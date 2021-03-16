//! Key structures for Orchard.

use std::convert::TryInto;
use std::mem;

use aes::Aes256;
use fpe::ff1::{BinaryNumeralString, FF1};
use group::GroupEncoding;
use halo2::{arithmetic::FieldExt, pasta::pallas};
use subtle::{Choice, CtOption};

use crate::{
    address::Address,
    primitives::redpallas::{self, SpendAuth},
    spec::{
        commit_ivk, diversify_hash, extract_p, ka_orchard, prf_expand, prf_expand_vec, prf_nf,
        to_base, to_scalar,
    },
};

/// A spending key, from which all key material is derived.
///
/// Defined in [Zcash Protocol Spec § 4.2.3: Orchard Key Components][orchardkeycomponents].
///
/// [orchardkeycomponents]: https://zips.z.cash/protocol/nu5.pdf#orchardkeycomponents
#[derive(Debug)]
pub struct SpendingKey([u8; 32]);

impl SpendingKey {
    /// Constructs an Orchard spending key from uniformly-random bytes.
    ///
    /// Returns `None` if the bytes do not correspond to a valid Orchard spending key.
    pub fn from_bytes(sk: [u8; 32]) -> CtOption<Self> {
        let sk = SpendingKey(sk);
        // If ask = 0, or the default address would be ⊥, discard this key.
        let ask_not_zero = !SpendAuthorizingKey::derive_inner(&sk).ct_is_zero();
        let have_default_address = Choice::from({
            let fvk = FullViewingKey::from(&sk);
            if fvk.default_address_inner().is_some() {
                1
            } else {
                0
            }
        });
        CtOption::new(sk, ask_not_zero & have_default_address)
    }
}

/// A spend authorizing key, used to create spend authorization signatures.
///
/// Defined in [Zcash Protocol Spec § 4.2.3: Orchard Key Components][orchardkeycomponents].
///
/// [orchardkeycomponents]: https://zips.z.cash/protocol/nu5.pdf#orchardkeycomponents
#[derive(Debug)]
pub(crate) struct SpendAuthorizingKey(redpallas::SigningKey<SpendAuth>);

impl SpendAuthorizingKey {
    /// Derives ask from sk. Internal use only, does not enforce all constraints.
    fn derive_inner(sk: &SpendingKey) -> pallas::Scalar {
        to_scalar(prf_expand(&sk.0, &[0x06]))
    }
}

impl From<&SpendingKey> for SpendAuthorizingKey {
    fn from(sk: &SpendingKey) -> Self {
        let ask = Self::derive_inner(sk);
        // SpendingKey cannot be constructed such that this assertion would fail.
        assert!(!bool::from(ask.ct_is_zero()));
        // TODO: Add TryFrom<S::Scalar> for SpendAuthorizingKey.
        let ret = SpendAuthorizingKey(ask.to_bytes().try_into().unwrap());
        // If the last bit of repr_P(ak) is 1, negate ask.
        if (<[u8; 32]>::from(SpendValidatingKey::from(&ret).0)[31] >> 7) == 1 {
            SpendAuthorizingKey((-ask).to_bytes().try_into().unwrap())
        } else {
            ret
        }
    }
}

/// A key used to validate spend authorization signatures.
///
/// Defined in [Zcash Protocol Spec § 4.2.3: Orchard Key Components][orchardkeycomponents].
/// Note that this is $\mathsf{ak}^\mathbb{P}$, which by construction is equivalent to
/// $\mathsf{ak}$ but stored here as a RedPallas verification key.
///
/// [orchardkeycomponents]: https://zips.z.cash/protocol/nu5.pdf#orchardkeycomponents
#[derive(Debug)]
pub(crate) struct SpendValidatingKey(redpallas::VerificationKey<SpendAuth>);

impl From<&SpendAuthorizingKey> for SpendValidatingKey {
    fn from(ask: &SpendAuthorizingKey) -> Self {
        SpendValidatingKey((&ask.0).into())
    }
}

/// A key used to derive [`Nullifier`]s from [`Note`]s.
///
/// Defined in [Zcash Protocol Spec § 4.2.3: Orchard Key Components][orchardkeycomponents].
///
/// [`Nullifier`]: crate::note::Nullifier
/// [`Note`]: crate::note::Note
/// [orchardkeycomponents]: https://zips.z.cash/protocol/nu5.pdf#orchardkeycomponents
#[derive(Debug)]
pub(crate) struct NullifierDerivingKey(pallas::Base);

impl From<&SpendingKey> for NullifierDerivingKey {
    fn from(sk: &SpendingKey) -> Self {
        NullifierDerivingKey(to_base(prf_expand(&sk.0, &[0x07])))
    }
}

impl NullifierDerivingKey {
    pub(crate) fn prf_nf(&self, rho: pallas::Base) -> pallas::Base {
        prf_nf(self.0, rho)
    }
}

/// The randomness for $\mathsf{Commit}^\mathsf{ivk}$.
///
/// Defined in [Zcash Protocol Spec § 4.2.3: Orchard Key Components][orchardkeycomponents].
///
/// [orchardkeycomponents]: https://zips.z.cash/protocol/nu5.pdf#orchardkeycomponents
#[derive(Debug)]
struct CommitIvkRandomness(pallas::Scalar);

impl From<&SpendingKey> for CommitIvkRandomness {
    fn from(sk: &SpendingKey) -> Self {
        CommitIvkRandomness(to_scalar(prf_expand(&sk.0, &[0x08])))
    }
}

/// A key that provides the capability to view incoming and outgoing transactions.
///
/// This key is useful anywhere you need to maintain accurate balance, but do not want the
/// ability to spend funds (such as a view-only wallet).
///
/// Defined in [Zcash Protocol Spec § 4.2.3: Orchard Key Components][orchardkeycomponents].
///
/// [orchardkeycomponents]: https://zips.z.cash/protocol/nu5.pdf#orchardkeycomponents
#[derive(Debug)]
pub struct FullViewingKey {
    ak: SpendValidatingKey,
    nk: NullifierDerivingKey,
    rivk: CommitIvkRandomness,
}

impl From<&SpendingKey> for FullViewingKey {
    fn from(sk: &SpendingKey) -> Self {
        FullViewingKey {
            ak: (&SpendAuthorizingKey::from(sk)).into(),
            nk: sk.into(),
            rivk: sk.into(),
        }
    }
}

impl FullViewingKey {
    pub(crate) fn nk(&self) -> &NullifierDerivingKey {
        &self.nk
    }

    /// Defined in [Zcash Protocol Spec § 4.2.3: Orchard Key Components][orchardkeycomponents].
    ///
    /// [orchardkeycomponents]: https://zips.z.cash/protocol/nu5.pdf#orchardkeycomponents
    fn derive_dk_ovk(&self) -> (DiversifierKey, OutgoingViewingKey) {
        let k = self.rivk.0.to_bytes();
        let b = [(&self.ak.0).into(), self.nk.0.to_bytes()];
        let r = prf_expand_vec(&k, &[&[0x82], &b[0][..], &b[1][..]]);
        (
            DiversifierKey(r[..32].try_into().unwrap()),
            OutgoingViewingKey(r[32..].try_into().unwrap()),
        )
    }

    /// Returns the default payment address for this key.
    pub fn default_address(&self) -> Address {
        self.default_address_inner()
            .expect("Default address works by construction")
    }

    fn default_address_inner(&self) -> Option<Address> {
        self.address(DiversifierKey::from(self).default_diversifier())
    }

    /// Returns the payment address for this key at the given index.
    ///
    /// Returns `None` if the diversifier does not correspond to an address. This happens
    /// with negligible probability; in most cases unwrapping the result will be fine, but
    /// if you have specific stability requirements then you can either convert this into
    /// an error, or try another diversifier index (e.g. incrementing).
    pub fn address_at(&self, j: impl Into<DiversifierIndex>) -> Option<Address> {
        self.address(DiversifierKey::from(self).get(j))
    }

    /// Returns the payment address for this key corresponding to the given diversifier.
    ///
    /// Returns `None` if the diversifier does not correspond to an address. This happens
    /// with negligible probability; in most cases unwrapping the result will be fine, but
    /// if you have specific stability requirements then you can either convert this into
    /// an error, or try another diversifier.
    pub fn address(&self, d: Diversifier) -> Option<Address> {
        IncomingViewingKey::from(self).address(d)
    }
}

/// A key that provides the capability to derive a sequence of diversifiers.
///
/// Defined in [Zcash Protocol Spec § 4.2.3: Orchard Key Components][orchardkeycomponents].
///
/// [orchardkeycomponents]: https://zips.z.cash/protocol/nu5.pdf#orchardkeycomponents
#[derive(Debug)]
pub struct DiversifierKey([u8; 32]);

impl From<&FullViewingKey> for DiversifierKey {
    fn from(fvk: &FullViewingKey) -> Self {
        fvk.derive_dk_ovk().0
    }
}

/// The index for a particular diversifier.
#[derive(Clone, Copy, Debug)]
pub struct DiversifierIndex([u8; 11]);

macro_rules! di_from {
    ($n:ident) => {
        impl From<$n> for DiversifierIndex {
            fn from(j: $n) -> Self {
                let mut j_bytes = [0; 11];
                j_bytes[..mem::size_of::<$n>()].copy_from_slice(&j.to_le_bytes());
                DiversifierIndex(j_bytes)
            }
        }
    };
}
di_from!(u32);
di_from!(u64);
di_from!(usize);

impl DiversifierKey {
    /// Returns the diversifier at index 0.
    pub fn default_diversifier(&self) -> Diversifier {
        self.get(0u32)
    }

    /// Returns the diversifier at the given index.
    pub fn get(&self, j: impl Into<DiversifierIndex>) -> Diversifier {
        let ff = FF1::<Aes256>::new(&self.0, 2).expect("valid radix");
        let enc = ff
            .encrypt(&[], &BinaryNumeralString::from_bytes_le(&j.into().0[..]))
            .unwrap();
        Diversifier(enc.to_bytes_le().try_into().unwrap())
    }
}

/// A diversifier that can be used to derive a specific [`Address`] from a
/// [`FullViewingKey`] or [`IncomingViewingKey`].
///
/// Defined in [Zcash Protocol Spec § 4.2.3: Orchard Key Components][orchardkeycomponents].
///
/// [orchardkeycomponents]: https://zips.z.cash/protocol/nu5.pdf#orchardkeycomponents
#[derive(Debug)]
pub struct Diversifier([u8; 11]);

impl Diversifier {
    /// Returns the byte array corresponding to this diversifier.
    pub fn as_array(&self) -> &[u8; 11] {
        &self.0
    }
}

/// A key that provides the capability to detect and decrypt incoming notes from the block
/// chain, without being able to spend the notes or detect when they are spent.
///
/// This key is useful in situations where you only need the capability to detect inbound
/// payments, such as merchant terminals.
///
/// This key is not suitable for use on its own in a wallet, as it cannot maintain
/// accurate balance. You should use a [`FullViewingKey`] instead.
///
/// Defined in [Zcash Protocol Spec § 4.2.3: Orchard Key Components][orchardkeycomponents].
///
/// [orchardkeycomponents]: https://zips.z.cash/protocol/nu5.pdf#orchardkeycomponents
#[derive(Debug)]
pub struct IncomingViewingKey(pallas::Scalar);

impl From<&FullViewingKey> for IncomingViewingKey {
    fn from(fvk: &FullViewingKey) -> Self {
        let ak = extract_p(&pallas::Point::from_bytes(&(&fvk.ak.0).into()).unwrap());
        IncomingViewingKey(commit_ivk(&ak, &fvk.nk.0, &fvk.rivk.0))
    }
}

impl IncomingViewingKey {
    /// Returns the payment address for this key corresponding to the given diversifier.
    ///
    /// Returns `None` if the diversifier does not correspond to an address. This happens
    /// with negligible probability; in most cases unwrapping the result will be fine, but
    /// if you have specific stability requirements then you can either convert this into
    /// an error, or try another diversifier.
    pub fn address(&self, d: Diversifier) -> Option<Address> {
        DiversifiedTransmissionKey::derive(self, &d).map(|pk_d| Address::from_parts(d, pk_d))
    }
}

/// A key that provides the capability to recover outgoing transaction information from
/// the block chain.
///
/// This key is not suitable for use on its own in a wallet, as it cannot maintain
/// accurate balance. You should use a [`FullViewingKey`] instead.
///
/// Defined in [Zcash Protocol Spec § 4.2.3: Orchard Key Components][orchardkeycomponents].
///
/// [orchardkeycomponents]: https://zips.z.cash/protocol/nu5.pdf#orchardkeycomponents
#[derive(Debug)]
pub struct OutgoingViewingKey([u8; 32]);

impl From<&FullViewingKey> for OutgoingViewingKey {
    fn from(fvk: &FullViewingKey) -> Self {
        fvk.derive_dk_ovk().1
    }
}

/// The diversified transmission key for a given payment address.
///
/// Defined in [Zcash Protocol Spec § 4.2.3: Orchard Key Components][orchardkeycomponents].
///
/// [orchardkeycomponents]: https://zips.z.cash/protocol/nu5.pdf#orchardkeycomponents
#[derive(Debug)]
pub(crate) struct DiversifiedTransmissionKey(pallas::Point);

impl DiversifiedTransmissionKey {
    /// Defined in [Zcash Protocol Spec § 4.2.3: Orchard Key Components][orchardkeycomponents].
    ///
    /// [orchardkeycomponents]: https://zips.z.cash/protocol/nu5.pdf#orchardkeycomponents
    fn derive(ivk: &IncomingViewingKey, d: &Diversifier) -> Option<Self> {
        diversify_hash(d.as_array()).map(|g_d| DiversifiedTransmissionKey(ka_orchard(&ivk.0, &g_d)))
    }

    /// $repr_P(self)$
    pub(crate) fn to_bytes(&self) -> [u8; 32] {
        self.0.to_bytes()
    }
}
