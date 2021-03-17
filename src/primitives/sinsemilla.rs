//! The Sinsemilla hash function and commitment scheme.

use group::Group;
use halo2::{arithmetic::CurveExt, pasta::pallas};

use crate::spec::extract_p;

mod constants;
pub use constants::*;

fn lebs2ip_k(bits: &[bool]) -> u32 {
    assert!(bits.len() == K);
    bits.iter()
        .enumerate()
        .fold(0u32, |acc, (i, b)| acc + if *b { 1 << i } else { 0 })
}

struct Pad<I: Iterator<Item = bool>> {
    inner: I,
    len: usize,
    padding_left: Option<usize>,
}

impl<I: Iterator<Item = bool>> Pad<I> {
    fn new(inner: I) -> Self {
        Pad {
            inner,
            len: 0,
            padding_left: None,
        }
    }
}

impl<I: Iterator<Item = bool>> Iterator for Pad<I> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(n) = self.padding_left.as_mut() {
                if *n == 0 {
                    break None;
                } else {
                    *n -= 1;
                    break Some(false);
                }
            } else if let Some(ret) = self.inner.next() {
                self.len += 1;
                assert!(self.len <= K * C);
                break Some(ret);
            } else {
                // Inner iterator just ended.
                let rem = self.len % K;
                if rem > 0 {
                    self.padding_left = Some(K - rem);
                } else {
                    // No padding required.
                    self.padding_left = Some(0);
                }
            }
        }
    }
}

pub trait HashDomain {
    #[allow(non_snake_case)]
    fn Q(&self) -> pallas::Point;
}

pub struct Domain(pub &'static str);

impl HashDomain for Domain {
    fn Q(&self) -> pallas::Point {
        pallas::Point::hash_to_curve(Q_PERSONALIZATION)(self.0.as_bytes())
    }
}

/// `SinsemillaHashToPoint` from [§ 5.4.1.9][concretesinsemillahash].
///
/// [concretesinsemillahash]: https://zips.z.cash/protocol/nu5.pdf#concretesinsemillahash
#[allow(non_snake_case)]
pub(crate) fn hash_to_point<D: HashDomain>(
    domain: &D,
    msg: impl Iterator<Item = bool>,
) -> pallas::Point {
    let padded: Vec<_> = Pad::new(msg).collect();

    let hasher_S = pallas::Point::hash_to_curve(S_PERSONALIZATION);
    let S = |chunk: &[bool]| hasher_S(&lebs2ip_k(chunk).to_le_bytes());

    padded
        .chunks(K)
        .fold(domain.Q(), |acc, chunk| acc.double() + S(chunk))
}

/// `SinsemillaHash` from [§ 5.4.1.9][concretesinsemillahash].
///
/// [concretesinsemillahash]: https://zips.z.cash/protocol/nu5.pdf#concretesinsemillahash
pub(crate) fn hash<D: HashDomain>(domain: &D, msg: impl Iterator<Item = bool>) -> pallas::Base {
    extract_p(&hash_to_point(domain, msg))
}

pub trait CommitDomain: HashDomain {
    #[allow(non_snake_case)]
    fn R(&self) -> pallas::Point;
}

pub struct Comm(pub &'static str);

impl HashDomain for Comm {
    fn Q(&self) -> pallas::Point {
        let m_prefix = self.0.to_owned() + "-M";
        pallas::Point::hash_to_curve(Q_PERSONALIZATION)(m_prefix.as_bytes())
    }
}

impl CommitDomain for Comm {
    fn R(&self) -> pallas::Point {
        let r_prefix = self.0.to_owned() + "-r";
        let hasher_r = pallas::Point::hash_to_curve(&r_prefix);
        hasher_r(&[])
    }
}

/// `SinsemillaCommit` from [§ 5.4.7.4][concretesinsemillacommit].
///
/// [concretesinsemillacommit]: https://zips.z.cash/protocol/nu5.pdf#concretesinsemillacommit
#[allow(non_snake_case)]
pub(crate) fn commit<D: CommitDomain>(
    domain: &D,
    msg: impl Iterator<Item = bool>,
    r: &pallas::Scalar,
) -> pallas::Point {
    hash_to_point(domain, msg) + domain.R() * r
}

/// `SinsemillaShortCommit` from [§ 5.4.7.4][concretesinsemillacommit].
///
/// [concretesinsemillacommit]: https://zips.z.cash/protocol/nu5.pdf#concretesinsemillacommit
pub(crate) fn short_commit<D: CommitDomain>(
    domain: &D,
    msg: impl Iterator<Item = bool>,
    r: &pallas::Scalar,
) -> pallas::Base {
    extract_p(&commit(domain, msg, r))
}

#[cfg(test)]
mod tests {
    use super::Pad;

    #[test]
    fn pad() {
        assert_eq!(Pad::new([].iter().cloned()).collect::<Vec<_>>(), vec![]);
        assert_eq!(
            Pad::new([true].iter().cloned()).collect::<Vec<_>>(),
            vec![true, false, false, false, false, false, false, false, false, false]
        );
        assert_eq!(
            Pad::new([true, true].iter().cloned()).collect::<Vec<_>>(),
            vec![true, true, false, false, false, false, false, false, false, false]
        );
        assert_eq!(
            Pad::new([true, true, true].iter().cloned()).collect::<Vec<_>>(),
            vec![true, true, true, false, false, false, false, false, false, false]
        );
        assert_eq!(
            Pad::new(
                [true, true, false, true, false, true, false, true, false, true]
                    .iter()
                    .cloned()
            )
            .collect::<Vec<_>>(),
            vec![true, true, false, true, false, true, false, true, false, true]
        );
        assert_eq!(
            Pad::new(
                [true, true, false, true, false, true, false, true, false, true, true]
                    .iter()
                    .cloned()
            )
            .collect::<Vec<_>>(),
            vec![
                true, true, false, true, false, true, false, true, false, true, true, false, false,
                false, false, false, false, false, false, false
            ]
        );
    }
}
