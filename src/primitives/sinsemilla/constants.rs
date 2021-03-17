//! Sinsemilla generators
use halo2::{
    arithmetic::{CurveAffine, CurveExt},
    pasta::pallas,
};

/// Number of bits of each message piece in SinsemillaHashToPoint
pub const K: usize = 10;

/// The largest integer such that 2^c <= (r_P - 1) / 2, where r_P is the order
/// of Pallas.
pub const C: usize = 253;

// Sinsemilla Q generators
/// SWU hash-to-curve personalization for Sinsemilla Q generators
pub const Q_PERSONALIZATION: &str = "z.cash:SinsemillaQ";

/// Generator used in SinsemillaHashToPoint for note commitment
pub const Q_NOTE_COMMITMENT_M_GENERATOR: (pallas::Base, pallas::Base, pallas::Base) = (
    pallas::Base::from_raw([
        0xcb32_150b_bd42_167f,
        0x0d67_fa10_5c39_d544,
        0xfc17_b491_75e9_5464,
        0x372e_2f74_da30_0868,
    ]),
    pallas::Base::from_raw([
        0x1156_abe0_9e25_60b6,
        0xbce9_ca0e_3950_8035,
        0x75e1_f360_d55a_fa25,
        0x3b75_a3b1_6173_a8d7,
    ]),
    pallas::Base::from_raw([
        0xdbc8_2388_468b_ac91,
        0xc81b_f7f1_7f13_b769,
        0x502e_6f16_dd62_7ff8,
        0x2229_42f2_0a34_9ca1,
    ]),
);

/// Generator used in SinsemillaHashToPoint for IVK commitment
pub const Q_COMMIT_IVK_M_GENERATOR: (pallas::Base, pallas::Base, pallas::Base) = (
    pallas::Base::from_raw([
        0xf8c3_aa6a_5398_bda3,
        0xc7b4_088e_c568_f566,
        0x2bea_4757_25b9_3595,
        0x0b5d_a9af_fb31_c540,
    ]),
    pallas::Base::from_raw([
        0x8fec_5cdf_3c96_10ff,
        0xec19_8983_4aa5_dc12,
        0x52f7_002c_08f7_9661,
        0x0333_96c7_3e0f_b9d8,
    ]),
    pallas::Base::from_raw([
        0xfa62_2168_f5cd_0aac,
        0xecf3_523b_842d_be49,
        0x995a_92f4_a214_b6f7,
        0x32ef_ef9b_fd54_f4a1,
    ]),
);

// Sinsemilla S generators

/// SWU hash-to-curve personalization for Sinsemilla S generators
pub const S_PERSONALIZATION: &str = "z.cash:SinsemillaS";

/// Creates the Sinsemilla S generators used in each round of the Sinsemilla hash
// TODO: inline the Sinsemilla S generators used in each round of the Sinsemilla hash
pub fn sinsemilla_s_generators<C: CurveAffine>() -> Vec<C::CurveExt> {
    let hasher = C::CurveExt::hash_to_curve(S_PERSONALIZATION);
    (0..(1 << K))
        .map(|j| hasher(&(j as usize).to_le_bytes()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::{Comm, HashDomain};
    use super::*;
    use crate::constants::{COMMIT_IVK_PERSONALIZATION, NOTE_COMMITMENT_PERSONALIZATION};
    use halo2::arithmetic::CurveExt;

    #[test]
    fn q_note_commitment_m() {
        let comm = Comm(NOTE_COMMITMENT_PERSONALIZATION);
        let point = comm.Q();
        let (x, y, z) = point.jacobian_coordinates();

        assert_eq!(x, Q_NOTE_COMMITMENT_M_GENERATOR.0);
        assert_eq!(y, Q_NOTE_COMMITMENT_M_GENERATOR.1);
        assert_eq!(z, Q_NOTE_COMMITMENT_M_GENERATOR.2);
    }

    #[test]
    fn q_commit_ivk_m() {
        let comm = Comm(COMMIT_IVK_PERSONALIZATION);
        let point = comm.Q();
        let (x, y, z) = point.jacobian_coordinates();

        assert_eq!(x, Q_COMMIT_IVK_M_GENERATOR.0);
        assert_eq!(y, Q_COMMIT_IVK_M_GENERATOR.1);
        assert_eq!(z, Q_COMMIT_IVK_M_GENERATOR.2);
    }
}
