//! Constants used in the Orchard protocol.
use ff::{Field, PrimeField};
use group::Curve;
use halo2::{
    arithmetic::{CurveAffine, CurveExt, FieldExt, Group},
    pasta::{pallas, Ep, EpAffine, Fp, Fq},
};

/// $\ell^\mathsf{Orchard}_\mathsf{base}$
pub(crate) const L_ORCHARD_BASE: usize = 255;

// SWU hash-to-curve personalizations

/// SWU hash-to-curve personalization for the spending key base point
pub const SPENDING_KEY_GENERATOR_PERSONALIZATION: &str = "z.cash:Orchard";

/// SWU hash-to-curve personalization for the group hash for key diversification
pub const KEY_DIVERSIFICATION_PERSONALIZATION: &str = "z.cash:Orchard-gd";

/// SWU hash-to-curve personalization for the value commitment generator
pub const VALUE_COMMITMENT_PERSONALIZATION: &str = "z.cash:Orchard-cv";

/// SWU hash-to-curve personalization for the note commitment generator
pub const NOTE_COMMITMENT_PERSONALIZATION: &str = "z.cash:Orchard-NoteCommit";

/// SWU hash-to-curve personalization for the IVK commitment generator
pub const COMMIT_IVK_PERSONALIZATION: &str = "z.cash:Orchard-CommitIvk";

/// SWU hash-to-curve personalization for the nullifier generator
pub const NULLIFIER_K_PERSONALIZATION: &str = "z.cash:Orchard-Nullifier-K";

// Sinsemilla commitment fixed generators
/// Generator used in SinsemillaCommit randomness for note commitment
pub const NOTE_COMMITMENT_R_GENERATOR: (pallas::Base, pallas::Base, pallas::Base) = (
    pallas::Base::from_raw([
        0x82ca_6040_b0f0_6a29,
        0x11dd_84e7_bb55_50c1,
        0x787e_8a8b_9462_53b9,
        0x3e26_1305_eb16_d974,
    ]),
    pallas::Base::from_raw([
        0xb750_aff4_8f11_a73d,
        0xc31b_4676_17e9_bef9,
        0x924d_5cf4_6ad8_d4a0,
        0x2605_fc3e_21a0_39ec,
    ]),
    pallas::Base::from_raw([
        0xaf01_5a86_7649_c5c0,
        0xed12_3064_feb3_3b59,
        0x085f_7f31_5ec3_6ff9,
        0x29e1_8d42_5f44_9d47,
    ]),
);

/// Generator used in SinsemillaCommit randomness for IVK commitment
pub const COMMIT_IVK_R_GENERATOR: (pallas::Base, pallas::Base, pallas::Base) = (
    pallas::Base::from_raw([
        0x3d96_5442_57f8_0d5a,
        0xc28c_0b1e_d367_c17c,
        0x0efd_1ef2_ec6a_2c00,
        0x1ae7_c99b_e3c6_2340,
    ]),
    pallas::Base::from_raw([
        0x9060_f9cc_f518_157c,
        0x95ef_0990_bc75_3050,
        0xf383_5cbb_b632_14c0,
        0x0bae_441b_1f81_8af9,
    ]),
    pallas::Base::from_raw([
        0x4978_fb2c_3756_fa42,
        0xad5e_a5a2_e0fe_ce89,
        0xcdf9_d9e6_e22d_a451,
        0x1eb3_f5fd_731a_b852,
    ]),
);

/// The value commitment is used to check balance between inputs and outputs. The value is
/// placed over this generator.
pub const VALUE_COMMITMENT_VALUE_GENERATOR: (pallas::Base, pallas::Base, pallas::Base) = (
    pallas::Base::from_raw([
        0xbe15_c28c_8a3c_3478,
        0xf7ff_1d10_23aa_7332,
        0xffdc_767b_1fa3_dadd,
        0x147b_2d2f_ce84_32ec,
    ]),
    pallas::Base::from_raw([
        0xd748_18a0_312b_874a,
        0x5919_5a13_371a_a8ed,
        0x9915_30a7_cdc7_0747,
        0x1957_39e6_9bdf_24cc,
    ]),
    pallas::Base::from_raw([
        0x9494_d6f0_18fa_986c,
        0x9e4d_6205_5bef_4803,
        0x37dd_ff6b_62d7_d89f,
        0x3245_140b_93be_8526,
    ]),
);

/// The value commitment is randomized over this generator, for privacy.
pub const VALUE_COMMITMENT_RANDOMNESS_GENERATOR: (pallas::Base, pallas::Base, pallas::Base) = (
    pallas::Base::from_raw([
        0xe6b7_1c0b_4ea3_f5db,
        0x9032_2bc4_a9b9_1717,
        0x4853_8459_9fc2_f119,
        0x22fe_9197_16ab_14a4,
    ]),
    pallas::Base::from_raw([
        0x4722_ccd3_af4f_e21e,
        0xfe26_3a68_50ac_86ee,
        0x3857_089b_ea98_4b96,
        0x060a_39c2_e7b2_0816,
    ]),
    pallas::Base::from_raw([
        0xcabd_2e60_e456_f64a,
        0x5faa_93d0_84d0_14a7,
        0x9bf2_9c55_90e6_2547,
        0x3131_216a_1a92_140d,
    ]),
);

/// Nullifier K^Orchard
pub const NULLIFIER_K_GENERATOR: (pallas::Base, pallas::Base, pallas::Base) = (
    pallas::Base::from_raw([
        0x9d5f_f764_3fba_eeb7,
        0xc372_539b_f87f_51be,
        0xa855_5e66_e662_4913,
        0x07b9_ce1c_b094_aea2,
    ]),
    pallas::Base::from_raw([
        0xa50b_6550_3480_7b49,
        0x280f_e9fb_f3b2_107b,
        0x282b_2d1f_9d3f_f02e,
        0x06cf_3110_209c_abd6,
    ]),
    pallas::Base::from_raw([
        0xc62e_df50_4f16_0366,
        0x2664_bd76_7dfe_5eea,
        0x5e62_bcf7_090f_180d,
        0x2a9d_5306_3aaa_e14d,
    ]),
);

/// Window size for fixed-base scalar multiplication
pub const FIXED_BASE_WINDOW_SIZE: usize = 3;

/// Number of windows
pub const NUM_WINDOWS: usize = Fp::NUM_BITS as usize / FIXED_BASE_WINDOW_SIZE;

fn compute_window_table(generator: pallas::Point) -> Vec<Vec<pallas::Point>> {
    let h: usize = 1 << FIXED_BASE_WINDOW_SIZE;
    let mut window_table: Vec<Vec<pallas::Point>> = Vec::with_capacity(NUM_WINDOWS);

    // Generate window table entries for all windows but the last
    for w in 0..(NUM_WINDOWS - 1) {
        window_table.push(
            (0..h)
                .map(|k| {
                    let scalar = Fq::from_u64(((k + 1) * h.pow(w as u32)) as u64);
                    let mut point = generator;
                    point.group_scale(&scalar);
                    point
                })
                .collect(),
        );
    }

    // Generate window table entries for the last window
    let sum = (0..(NUM_WINDOWS - 1)).fold(0, |acc, w| acc + h.pow(w as u32));
    window_table.push(
        (0..h)
            .map(|k| {
                let scalar = Fq::from_u64((k * h.pow((NUM_WINDOWS - 1) as u32) - sum) as u64);
                let mut point = generator;
                point.group_scale(&scalar);
                point
            })
            .collect(),
    );

    window_table
}

// TODO: write window tables for each generator to .dat files?
/// Window table for the note commitment randomness generator
pub fn window_table_note_commitment_r() -> Vec<Vec<pallas::Point>> {
    let generator = Ep::new_jacobian(
        NOTE_COMMITMENT_R_GENERATOR.0,
        NOTE_COMMITMENT_R_GENERATOR.1,
        NOTE_COMMITMENT_R_GENERATOR.2,
    )
    .unwrap();
    compute_window_table(generator)
}

/// Window table for the commit_ivk randomness generator
pub fn window_table_commit_ivk_r() -> Vec<Vec<pallas::Point>> {
    let generator = Ep::new_jacobian(
        COMMIT_IVK_R_GENERATOR.0,
        COMMIT_IVK_R_GENERATOR.1,
        COMMIT_IVK_R_GENERATOR.2,
    )
    .unwrap();
    compute_window_table(generator)
}

/// Window table for the value commitment value generator
pub fn window_table_value_commit_v() -> Vec<Vec<pallas::Point>> {
    let generator = Ep::new_jacobian(
        VALUE_COMMITMENT_VALUE_GENERATOR.0,
        VALUE_COMMITMENT_VALUE_GENERATOR.1,
        VALUE_COMMITMENT_VALUE_GENERATOR.2,
    )
    .unwrap();
    compute_window_table(generator)
}

/// Window table for the value commitment randomness generator
pub fn window_table_value_commit_r() -> Vec<Vec<pallas::Point>> {
    let generator = Ep::new_jacobian(
        VALUE_COMMITMENT_RANDOMNESS_GENERATOR.0,
        VALUE_COMMITMENT_RANDOMNESS_GENERATOR.1,
        VALUE_COMMITMENT_RANDOMNESS_GENERATOR.2,
    )
    .unwrap();
    compute_window_table(generator)
}

/// Window table for the K^Orchard nullifier generator
pub fn window_table_nullifier_k() -> Vec<Vec<pallas::Point>> {
    let generator = Ep::new_jacobian(
        NULLIFIER_K_GENERATOR.0,
        NULLIFIER_K_GENERATOR.1,
        NULLIFIER_K_GENERATOR.2,
    )
    .unwrap();
    compute_window_table(generator)
}

fn find_z(window_points: &[pallas::Point]) -> Option<u64> {
    let h = 1 << FIXED_BASE_WINDOW_SIZE;
    assert_eq!(h, window_points.len());

    let mut window_points_affine = vec![EpAffine::default(); h];
    Ep::batch_normalize(window_points, &mut window_points_affine);

    let ys: Vec<_> = window_points_affine
        .iter()
        .map(|point| point.get_xy().unwrap().1)
        .collect();
    let z_for_single_y = |y: Fp, z: u64| {
        let sum_y_is_square: bool = (y + Fp::from_u64(z)).sqrt().is_some().into();
        let sum_neg_y_is_square: bool = (-y + Fp::from_u64(z)).sqrt().is_some().into();
        (sum_y_is_square && !sum_neg_y_is_square) as usize
    };

    for z in 0..(1000 * (1 << (2 * h))) {
        if ys.iter().map(|y| z_for_single_y(*y, z)).sum::<usize>() == h {
            return Some(z);
        }
    }

    None
}

// TODO: write z results to a .dat file?
/// z-values for the note commitment randomness generator
pub fn z_note_commitment_r() -> Vec<u64> {
    window_table_note_commitment_r()
        .iter()
        .map(|window_points| find_z(window_points).unwrap())
        .collect::<Vec<_>>()
}

/// z-values for the commit_ivk randomness generator
pub fn z_commit_ivk_r() -> Vec<u64> {
    window_table_commit_ivk_r()
        .iter()
        .map(|window_points| find_z(window_points).unwrap())
        .collect::<Vec<_>>()
}

/// z-values for the value commitment value generator
pub fn z_value_commit_v() -> Vec<u64> {
    window_table_value_commit_v()
        .iter()
        .map(|window_points| find_z(window_points).unwrap())
        .collect::<Vec<_>>()
}

/// z-values for the value commitment randomness generator
pub fn z_value_commit_r() -> Vec<u64> {
    window_table_value_commit_r()
        .iter()
        .map(|window_points| find_z(window_points).unwrap())
        .collect::<Vec<_>>()
}

/// z-values for the K^Orchard nullifier generator
pub fn z_nullifier_k() -> Vec<u64> {
    window_table_nullifier_k()
        .iter()
        .map(|window_points| find_z(window_points).unwrap())
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{COMMIT_IVK_PERSONALIZATION, NOTE_COMMITMENT_PERSONALIZATION};
    use halo2::{arithmetic::CurveExt, pasta::pallas};

    #[test]
    fn note_commitment_r() {
        let point = pallas::Point::hash_to_curve(
            format!("{}{}", NOTE_COMMITMENT_PERSONALIZATION, "-r").as_str(),
        )(b"");
        let (x, y, z) = point.jacobian_coordinates();

        assert_eq!(x, NOTE_COMMITMENT_R_GENERATOR.0);
        assert_eq!(y, NOTE_COMMITMENT_R_GENERATOR.1);
        assert_eq!(z, NOTE_COMMITMENT_R_GENERATOR.2);
    }

    #[test]
    fn commit_ivk_r() {
        let point = pallas::Point::hash_to_curve(
            format!("{}{}", COMMIT_IVK_PERSONALIZATION, "-r").as_str(),
        )(b"");
        let (x, y, z) = point.jacobian_coordinates();

        assert_eq!(x, COMMIT_IVK_R_GENERATOR.0);
        assert_eq!(y, COMMIT_IVK_R_GENERATOR.1);
        assert_eq!(z, COMMIT_IVK_R_GENERATOR.2);
    }
}
