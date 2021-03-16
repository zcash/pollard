use halo2::{arithmetic::CurveExt, pasta::pallas};

use super::NoteCommitment;
use crate::{
    keys::NullifierDerivingKey,
    spec::{extract_p, mod_r_p},
};

/// A unique nullifier for a note.
#[derive(Debug)]
pub struct Nullifier(pub(super) pallas::Base);

impl Nullifier {
    /// $DeriveNullifier$.
    ///
    /// Defined in [Zcash Protocol Spec § 4.16: Note Commitments and Nullifiers][commitmentsandnullifiers].
    ///
    /// [commitmentsandnullifiers]: https://zips.z.cash/protocol/nu5.pdf#commitmentsandnullifiers
    pub(super) fn derive(
        rho: pallas::Base,
        psi: pallas::Base,
        cm: NoteCommitment,
        nk: &NullifierDerivingKey,
    ) -> Self {
        let k = pallas::Point::hash_to_curve("TODO")(b"TODO");

        Nullifier(extract_p(&(k * mod_r_p(nk.prf_nf(rho) + psi) + cm.0)))
    }
}
