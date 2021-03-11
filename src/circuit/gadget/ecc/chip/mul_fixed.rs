use super::super::EccInstructions;
use super::{add, util, witness_point, EccChip, EccPoint};

use halo2::{
    arithmetic::{CurveAffine, Field, FieldExt},
    circuit::{Chip, Region},
    plonk::{Column, ConstraintSystem, Error, Expression, Fixed},
    poly::Rotation,
};

pub(super) fn create_gate<C: CurveAffine>(
    meta: &mut ConstraintSystem<C::Base>,
    number_base: usize,
    lagrange_coeffs: [Column<Fixed>; 8],
    q_mul_fixed: Expression<C::Base>,
    x_p: Expression<C::Base>,
    y_p: Expression<C::Base>,
    k: Expression<C::Base>,
    u: Expression<C::Base>,
    z: Expression<C::Base>,
) {
    // x-coordinate
    meta.create_gate("fixed-base scalar mul (x)", |meta| {
        let k_pow = (0..number_base).scan(Expression::Constant(C::Base::one()), |state, _| {
            *state = state.clone() * k.clone();
            Some(state.clone())
        });

        let interpolated_x = k_pow.zip(lagrange_coeffs.iter()).fold(
            Expression::Constant(C::Base::zero()),
            |acc, (k_pow, coeff)| acc + (k_pow * meta.query_fixed(*coeff, Rotation::cur())),
        );

        q_mul_fixed.clone() * (interpolated_x - x_p)
    });

    // y-coordinate
    meta.create_gate("fixed-base scalar mul (y)", |meta| {
        q_mul_fixed * (u.clone() * u - y_p - z)
    });
}

pub(super) fn assign_region<C: CurveAffine>(
    scalar: &<EccChip<C> as EccInstructions<C>>::ScalarFixed,
    base: &<EccChip<C> as EccInstructions<C>>::FixedPoint,
    number_base: C::Scalar,
    region: &mut Region<'_, EccChip<C>>,
    config: <EccChip<C> as Chip>::Config,
) -> Result<EccPoint<C::Base>, Error> {
    // Assign fixed columns for given fixed base
    for w in 0..config.num_windows {
        // Enable relevant selectors
        config.q_mul_fixed.enable(region, w)?;

        for k in 0..(1 << config.window_width) {
            // Assign x-coordinate Lagrange interpolation coefficients
            region.assign_fixed(
                || {
                    format!(
                        "Lagrange interpolation coefficient for window: {:?}, k: {:?}",
                        w, k
                    )
                },
                config.lagrange_coeffs[k],
                w,
                || {
                    base.lagrange_coeffs
                        .as_ref()
                        .map(|c| c[w][k])
                        .ok_or(Error::SynthesisError)
                },
            )?;
        }
        // Assign z-values for each window
        region.assign_fixed(
            || format!("z-value for window: {:?}", w),
            config.fixed_z,
            w,
            || base.z.as_ref().map(|z| z[w]).ok_or(Error::SynthesisError),
        )?;
    }

    // Cumulative variable used to decompose the scalar in-circuit.
    // m_n = k_n (MSB), m_0 = scalar
    // m_{i-1} = m_i * 8 + k_{i-1}
    // m = [m_n, m_{n-1}, ..., m_0] (little-endian)

    let b = base.fixed_point.value();
    let k = scalar
        .k_bits
        .iter()
        .map(|bits| C::Scalar::from_bytes(&(*bits).value.unwrap().to_bytes()).unwrap())
        .collect::<Vec<_>>();

    // Copy the scalar decomposition
    for (w, k) in scalar.k_bits.iter().enumerate() {
        util::assign_and_constrain(
            region,
            || format!("k[{:?}]", w),
            config.k,
            w,
            k,
            &config.perm_scalar,
        )?;
    }

    // Process the MSB outside the for loop
    let mul_b = b * k[0] * number_base;
    let mul_b = witness_point::assign_region(Some(mul_b), region, config.clone()).unwrap();

    // Compute u = y_p + z_w
    let u_val = mul_b.y.value.unwrap() + base.z.as_ref().unwrap()[0];
    region.assign_advice(|| "u", config.u, 0, || Ok(u_val))?;

    // Initialise the point which will cumulatively sum to [scalar]B
    // Copy and assign mul_b to x_a, y_a columns on the next row
    let x_sum = util::assign_and_constrain(
        region,
        || "initialize sum x",
        config.x_a,
        1,
        &mul_b.x,
        &config.perm_sum,
    )?;
    let y_sum = util::assign_and_constrain(
        region,
        || "initialize sum y",
        config.y_a,
        1,
        &mul_b.y,
        &config.perm_sum,
    )?;

    let mut sum = EccPoint { x: x_sum, y: y_sum };

    for ((w, k), z) in k
        .iter()
        .skip(1)
        .enumerate()
        .zip(base.z.as_ref().unwrap().iter().skip(1))
    {
        // Offset index by 1 since we assigned row 0 outside this for loop
        let w = w + 1;

        // Compute [k_w ⋅ 8^w]B
        let mul_b = b * *k * number_base.pow(&[w as u64, 0, 0, 0]);

        let mul_b = witness_point::assign_region(Some(mul_b), region, config.clone()).unwrap();

        // Compute u = y_p + z_w
        let u_val = mul_b.y.value.unwrap() + z;
        region.assign_advice(|| "u", config.u, w, || Ok(u_val))?;

        // Add to the cumulative sum
        sum = add::assign_region(&sum, &mul_b, region, config.clone()).unwrap();
    }

    Ok(sum)
}
