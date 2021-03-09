use super::{CellValue, EccChip, EccScalarFixed};

use ff::PrimeField;
use halo2::{
    arithmetic::{CurveAffine, Field, FieldExt},
    circuit::{Chip, Region},
    plonk::{ConstraintSystem, Error, Expression},
};

pub(super) fn create_gate<C: CurveAffine>(
    meta: &mut ConstraintSystem<C::Base>,
    number_base: usize,
    q_scalar_fixed: Expression<C::Base>,
    k: Expression<C::Base>,
) {
    meta.create_gate("witness point", |_| {
        // Check that 0 <= k <= 8
        let range_check = (0..(number_base + 1))
            .fold(Expression::Constant(C::Base::one()), |acc, i| {
                acc * (k.clone() - Expression::Constant(C::Base::from_u64(i as u64)))
            });
        q_scalar_fixed * range_check
    });
}

pub(super) fn assign_region<C: CurveAffine>(
    value: Option<C::Scalar>,
    region: &mut Region<'_, EccChip<C>>,
    config: <EccChip<C> as Chip>::Config,
) -> Result<EccScalarFixed<C>, Error> {
    // Scalar decomposed in three-bit windows (big-endian).
    // This is [k_n, ..., k_0], where each k_i is a 3-bit value and
    // scalar = k_n * 8^n + ... + k_1 * 8 + k_0.
    let bits: Option<Vec<u8>> = value
        .map(|value| {
            value
                .to_le_bits()
                .into_iter()
                .take(C::Scalar::NUM_BITS as usize)
                .collect()
        })
        .map(|bits: Vec<bool>| {
            assert_eq!(bits.len(), C::Scalar::NUM_BITS as usize);
            let mut bits: Vec<u8> = bits
                .chunks_exact(config.window_width)
                .map(|chunk| {
                    let mut chunk = chunk.iter();
                    *(chunk.next().unwrap()) as u8 + (*(chunk.next().unwrap()) as u8)
                        << 1 + (*(chunk.next().unwrap()) as u8)
                        << 2
                })
                .collect();
            bits.reverse();
            bits
        });

    let mut k_bits: Vec<CellValue<C::Base>> = Vec::new();

    if let Some(bits) = bits {
        for (idx, three_bits) in bits.iter().enumerate() {
            // Enable q_scalar_fixed selector
            config.q_scalar_fixed.enable(region, idx)?;

            let three_bits = C::Base::from_u64(*three_bits as u64);
            let k_var = region.assign_advice(
                || format!("k[{:?}]", idx),
                config.k,
                idx,
                || Ok(three_bits),
            )?;
            k_bits[idx] = CellValue::new(k_var, Some(three_bits));
        }
    }

    Ok(EccScalarFixed { value, k_bits })
}
