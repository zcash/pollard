use super::{CellValue, EccChip, EccPoint};

use group::Curve;
use halo2::{
    arithmetic::{CurveAffine, Field, FieldExt},
    circuit::{Chip, Region},
    plonk::{ConstraintSystem, Error, Expression},
};

pub(super) fn create_gate<C: CurveAffine>(
    meta: &mut ConstraintSystem<C::Base>,
    q_point: Expression<C::Base>,
    x_p: Expression<C::Base>,
    y_p: Expression<C::Base>,
) {
    meta.create_gate("witness point", |_| {
        // Check that y^2 = x^3 + b, where b = 5 in the Pallas equation
        q_point
            * (y_p.clone() * y_p
                - (x_p.clone() * x_p.clone() * x_p)
                - Expression::Constant(C::Base::from_u64(5)))
    });
}

pub(super) fn assign_region<C: CurveAffine>(
    value: Option<C::CurveExt>,
    region: &mut Region<'_, EccChip<C>>,
    config: <EccChip<C> as Chip>::Config,
) -> Result<EccPoint<C::Base>, Error> {
    let value = value.map(|value| value.to_affine().get_xy().unwrap());
    let x_p_val = value.map(|value| value.0);
    let y_p_val = value.map(|value| value.1);

    let x_p_var = region.assign_advice(
        || "x_p",
        config.x_p,
        0,
        || x_p_val.ok_or(Error::SynthesisError),
    )?;
    let y_p_var = region.assign_advice(
        || "y_p",
        config.y_p,
        0,
        || y_p_val.ok_or(Error::SynthesisError),
    )?;

    Ok(EccPoint {
        x: CellValue::new(x_p_var, x_p_val),
        y: CellValue::new(y_p_var, y_p_val),
    })
}
