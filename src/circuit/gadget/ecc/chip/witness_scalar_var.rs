use super::{CellValue, EccChip, EccScalarVar};

use ff::PrimeField;
use halo2::{
    arithmetic::{CurveAffine, Field, FieldExt},
    circuit::{Chip, Region},
    plonk::{ConstraintSystem, Error, Expression},
};

pub(super) fn create_gate<C: CurveAffine>(
    meta: &mut ConstraintSystem<C::Base>,
    q_scalar_var: Expression<C::Base>,
    k: Expression<C::Base>,
) {
    meta.create_gate("witness point", |_| {
        // Check that k \in {0, 1}
        q_scalar_var * (k.clone()) * (Expression::Constant(C::Base::one()) - k)
    });
}

pub(super) fn assign_region<C: CurveAffine>(
    value: Option<C::Scalar>,
    region: &mut Region<'_, EccChip<C>>,
    config: <EccChip<C> as Chip>::Config,
) -> Result<EccScalarVar<C>, Error> {
    // The scalar field F_q = 2^254 + t_q
    let t_q = u128::from_str_radix(C::Scalar::MODULUS, 16).unwrap() - (1 << C::Scalar::CAPACITY);

    // We will witness k = scalar + t_q
    // k is decomposed bitwise in-circuit for our double-and-add algorithm.
    let k = value.map(|value| value + C::Scalar::from_u128(t_q));

    // k decomposed bitwise (big-endian)
    // This is [k_n, ..., k_0], where each k_i is a bit and
    // l = k_n * 2^n + ... + k_1 * 2 + k_0.
    let bits: Option<Vec<bool>> = k.map(|k| {
        let mut bits: Vec<bool> = k
            .to_le_bits()
            .into_iter()
            .take(C::Scalar::NUM_BITS as usize)
            .collect();
        bits.reverse();
        bits
    });

    let mut k_bits: Vec<CellValue<C::Base>> = Vec::new();

    if let Some(bits) = bits {
        for (idx, bit) in bits.iter().enumerate() {
            // Enable q_scalar_var selector
            config.q_scalar_var.enable(region, idx)?;

            let bit = C::Base::from_u64(*bit as u64);
            let k_var =
                region.assign_advice(|| format!("k[{:?}]", idx), config.k, idx, || Ok(bit))?;
            k_bits[idx] = CellValue::new(k_var, Some(bit));
        }
    }

    Ok(EccScalarVar { value, k_bits })
}
