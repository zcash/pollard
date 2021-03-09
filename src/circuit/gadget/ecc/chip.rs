use std::{collections::HashMap, marker::PhantomData};

use super::{EccInstructions, FixedPoints};
use ff::Field;
use group::Curve;
use halo2::{
    arithmetic::{lagrange_interpolate, CurveAffine, CurveExt, FieldExt},
    circuit::{Cell, Chip, Layouter},
    plonk::{Advice, Column, ConstraintSystem, Error, Fixed, Permutation, Selector},
    poly::Rotation,
};

pub(crate) mod add;
mod double;
pub(crate) mod util;
pub(crate) mod witness_point;
pub(crate) mod witness_scalar_fixed;

/// Configuration for the ECC chip
#[derive(Clone, Debug)]
pub struct EccConfig {
    num_windows: usize,
    window_width: usize,
    num_complete_bits: usize,
    k: Column<Advice>,
    x_a: Column<Advice>,
    y_a: Column<Advice>,
    x_p: Column<Advice>,
    y_p: Column<Advice>,
    u: Column<Advice>,
    lambda1: Column<Advice>,
    lambda2: Column<Advice>,
    add_complete_inv: [Column<Advice>; 4],
    add_complete_bool: [Column<Advice>; 4],
    lagrange_coeffs: [Column<Fixed>; 8],
    fixed_z: Column<Fixed>,
    q_add: Selector,
    q_add_complete: Selector,
    q_double: Selector,
    q_mul: Selector,
    q_mul_fixed: Selector,
    q_point: Selector,
    q_scalar_var: Selector,
    q_scalar_fixed: Selector,
    perm_scalar: Permutation,
    perm_sum: Permutation,
}

/// A chip implementing EccInstructions
#[derive(Debug)]
pub struct EccChip<C: CurveAffine> {
    _marker: PhantomData<C>,
}

impl EccConfig {
    fn configure<C: CurveAffine>(
        meta: &mut ConstraintSystem<C::Base>,
        num_windows: usize,
        window_width: usize,
        num_complete_bits: usize,
        k: Column<Advice>,
        x_a: Column<Advice>,
        y_a: Column<Advice>,
        x_p: Column<Advice>,
        y_p: Column<Advice>,
        u: Column<Advice>,
        lambda1: Column<Advice>,
        lambda2: Column<Advice>,
        add_complete_inv: [Column<Advice>; 4],
        add_complete_bool: [Column<Advice>; 4],
    ) -> EccConfig {
        let number_base = 1 << window_width;

        let q_add = meta.selector();
        let q_add_complete = meta.selector();
        let q_double = meta.selector();
        let q_mul = meta.selector();
        let q_mul_fixed = meta.selector();
        let q_point = meta.selector();
        let q_scalar_var = meta.selector();
        let q_scalar_fixed = meta.selector();

        let lagrange_coeffs = [
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
        ];
        let fixed_z = meta.fixed_column();

        // Copy decomposed scalar
        let perm_scalar = Permutation::new(meta, &[k.into()]);

        // Initialise some accumulator from a witnessed point
        let perm_sum = Permutation::new(meta, &[x_p.into(), y_p.into(), x_a.into(), y_a.into()]);

        // Create witness point gate
        {
            let q_point = meta.query_selector(q_point, Rotation::cur());
            let x_p = meta.query_advice(x_p, Rotation::cur());
            let y_p = meta.query_advice(y_p, Rotation::cur());
            witness_point::create_gate::<C>(meta, q_point, x_p, y_p);
        }

        // Create witness scalar_fixed gate
        {
            let q_scalar_fixed = meta.query_selector(q_scalar_fixed, Rotation::cur());
            let k = meta.query_advice(k, Rotation::cur());
            witness_scalar_fixed::create_gate::<C>(meta, number_base, q_scalar_fixed, k);
        }

        // Create point doubling gate
        {
            let q_double = meta.query_selector(q_double, Rotation::cur());
            let x_a = meta.query_advice(x_a, Rotation::cur());
            let y_a = meta.query_advice(y_a, Rotation::cur());
            let x_p = meta.query_advice(x_p, Rotation::cur());
            let y_p = meta.query_advice(y_p, Rotation::cur());

            double::create_gate::<C>(meta, q_double, x_a, y_a, x_p, y_p);
        }

        // Create point addition gate
        {
            let q_add = meta.query_selector(q_add, Rotation::cur());
            let x_p = meta.query_advice(x_p, Rotation::cur());
            let y_p = meta.query_advice(y_p, Rotation::cur());
            let x_q = meta.query_advice(x_a, Rotation::cur());
            let y_q = meta.query_advice(y_a, Rotation::cur());
            let x_a = meta.query_advice(x_a, Rotation::next());
            let y_a = meta.query_advice(y_a, Rotation::next());

            add::create_gate::<C>(meta, q_add, x_p, y_p, x_q, y_q, x_a, y_a);
        }

        EccConfig {
            num_windows,
            window_width,
            num_complete_bits,
            k,
            x_a,
            y_a,
            x_p,
            y_p,
            u,
            lambda1,
            lambda2,
            add_complete_inv,
            add_complete_bool,
            lagrange_coeffs,
            fixed_z,
            q_add,
            q_add_complete,
            q_double,
            q_mul,
            q_mul_fixed,
            q_point,
            q_scalar_var,
            q_scalar_fixed,
            perm_scalar,
            perm_sum,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EccLoaded<C: CurveAffine> {
    lagrange_coeffs: Option<HashMap<OrchardFixedPoints<C>, Vec<Vec<C::Base>>>>,
    z: Option<HashMap<OrchardFixedPoints<C>, Vec<C::Base>>>,
}

impl<C: CurveAffine> EccLoaded<C> {
    fn get_lagrange_coeffs(&self, point: OrchardFixedPoints<C>) -> Option<Vec<Vec<C::Base>>> {
        if let Some(lagrange_coeffs) = &self.lagrange_coeffs {
            lagrange_coeffs.get(&point).cloned()
        } else {
            None
        }
    }

    fn get_z(&self, point: OrchardFixedPoints<C>) -> Option<Vec<C::Base>> {
        if let Some(z) = &self.z {
            z.get(&point).cloned()
        } else {
            None
        }
    }
}

impl<C: CurveAffine> Chip for EccChip<C> {
    type Config = EccConfig;
    type Field = C::Base;
    type Loaded = EccLoaded<C>;

    fn load(layouter: &mut impl Layouter<Self>) -> Result<Self::Loaded, Error> {
        let config = layouter.config().clone();
        let number_base = (1 as u64) << config.window_width;

        // Closure to compute multiples of a given fixed base B
        let get_matrix = |fixed_base: C::CurveExt| {
            // For the first 84 rows, M[w][k] = [(k+1)8^w]B
            let mut matrix_points = (0..(config.num_windows - 1))
                .map(|w| {
                    (0..number_base)
                        .map(|k| {
                            fixed_base
                                * C::Scalar::from_u64(number_base.pow(w as u32))
                                * C::Scalar::from_u64((k + 1) as u64)
                        })
                        .collect::<Vec<C::CurveExt>>()
                })
                .collect::<Vec<Vec<C::CurveExt>>>();

            // In the last row, M[w][k] = [(k)8^w - \sum\limits_{j=0}^{83} (8)^j]B
            let offset_sum = C::Scalar::from_u64(
                (0..(config.num_windows - 2)).fold(0, |acc, _| acc + number_base * acc),
            );
            matrix_points.push(
                (0..config.window_width)
                    .map(|k| {
                        let scalar =
                            C::Scalar::from_u64(number_base.pow(config.num_windows as u32 - 1))
                                * C::Scalar::from_u64(k as u64)
                                - offset_sum;
                        fixed_base * scalar
                    })
                    .collect::<Vec<C::CurveExt>>(),
            );
            let mut matrix = vec![C::default(); config.num_windows * config.window_width];
            C::Curve::batch_normalize(
                &matrix_points.into_iter().flatten().collect::<Vec<_>>(),
                &mut matrix,
            );
            matrix
                .iter()
                .map(|affine| affine.get_xy().unwrap())
                .collect::<Vec<_>>()
                .chunks_exact(config.window_width)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<_>>()
        };

        // Closure to compute interpolation polynomial coefficients for the x-coordinate
        let get_lagrange_coeffs = |matrix: Vec<Vec<(C::Base, C::Base)>>| {
            let points: Vec<C::Base> = (0..config.window_width)
                .map(|i| C::Base::from_u64(i as u64))
                .collect();
            matrix
                .iter()
                .map(|evals| {
                    let x_evals: Vec<_> = evals.iter().map(|eval| eval.0).collect();
                    lagrange_interpolate(&points, &x_evals)
                })
                .collect::<Vec<Vec<C::Base>>>()
        };

        // Closure to compute z for every row of the matrix
        let get_z = |matrix: Vec<Vec<(C::Base, C::Base)>>, h: usize| {
            // Closure to compute z for a single y-coordinate
            let find_single_z = |y: C::Base, start: u64| -> Option<u64> {
                for tmp_z in start..(1000 * (1 << (2 * h))) {
                    let z = C::Base::from_u64(tmp_z);
                    if (y + z).sqrt().is_some().into() && (-y + z).sqrt().is_none().into() {
                        return Some(tmp_z);
                    }
                }
                None
            };
            matrix
                .iter()
                .map(|evals| {
                    let y_evals: Vec<_> = evals.iter().map(|eval| eval.1).collect();
                    let mut z = 1u64;
                    for y in y_evals.iter() {
                        z = find_single_z(*y, z).unwrap();
                    }
                    C::Base::from_u64(z)
                })
                .collect::<Vec<C::Base>>()
        };

        let mut lagrange_coeffs = HashMap::<OrchardFixedPoints<C>, Vec<Vec<C::Base>>>::new();
        let mut z = HashMap::<OrchardFixedPoints<C>, Vec<C::Base>>::new();
        let h = 8;

        {
            let hasher = C::CurveExt::hash_to_curve("z.cash:Orchard-Nullifier-K");
            let nullifier_base = hasher(b"");
            let matrix = get_matrix(nullifier_base);
            lagrange_coeffs.insert(
                OrchardFixedPoints::NullifierBase(nullifier_base),
                get_lagrange_coeffs(matrix.clone()),
            );
            z.insert(
                OrchardFixedPoints::NullifierBase(nullifier_base),
                get_z(matrix, 8),
            );
        }
        {
            let hasher = C::CurveExt::hash_to_curve("z.cash:Orchard-NoteCommit-r");
            let note_commit_base = hasher(b"");
            let matrix = get_matrix(note_commit_base);
            lagrange_coeffs.insert(
                OrchardFixedPoints::NoteCommitBase(note_commit_base),
                get_lagrange_coeffs(matrix.clone()),
            );
            z.insert(
                OrchardFixedPoints::NoteCommitBase(note_commit_base),
                get_z(matrix, h),
            );
        }
        {
            let hasher = C::CurveExt::hash_to_curve("z.cash:Orchard-cv");
            let value_commit_v = hasher(b"v");
            let matrix = get_matrix(value_commit_v);
            lagrange_coeffs.insert(
                OrchardFixedPoints::ValueCommitV(value_commit_v),
                get_lagrange_coeffs(matrix.clone()),
            );
            z.insert(
                OrchardFixedPoints::ValueCommitV(value_commit_v),
                get_z(matrix, h),
            );
        }
        {
            let hasher = C::CurveExt::hash_to_curve("z.cash:Orchard-cv");
            let value_commit_r = hasher(b"r");
            let matrix = get_matrix(value_commit_r);
            lagrange_coeffs.insert(
                OrchardFixedPoints::ValueCommitR(value_commit_r),
                get_lagrange_coeffs(matrix.clone()),
            );
            z.insert(
                OrchardFixedPoints::ValueCommitR(value_commit_r),
                get_z(matrix, h),
            );
        }
        {
            let hasher = C::CurveExt::hash_to_curve("z.cash:Orchard-CommitIvk-r");
            let commit_ivk_base = hasher(b"");
            let matrix = get_matrix(commit_ivk_base);
            lagrange_coeffs.insert(
                OrchardFixedPoints::CommitIvkBase(commit_ivk_base),
                get_lagrange_coeffs(matrix.clone()),
            );
            z.insert(
                OrchardFixedPoints::CommitIvkBase(commit_ivk_base),
                get_z(matrix, h),
            );
        }

        Ok(EccLoaded {
            lagrange_coeffs: Some(lagrange_coeffs),
            z: Some(z),
        })
    }
}

// enum containing each fixed point type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrchardFixedPoints<C: CurveAffine> {
    NullifierBase(C::CurveExt),
    NoteCommitBase(C::CurveExt),
    ValueCommitV(C::CurveExt),
    ValueCommitR(C::CurveExt),
    CommitIvkBase(C::CurveExt),
}

impl<C: CurveAffine> OrchardFixedPoints<C> {
    fn value(&self) -> C::CurveExt {
        match *self {
            OrchardFixedPoints::NullifierBase(point) => point,
            OrchardFixedPoints::NoteCommitBase(point) => point,
            OrchardFixedPoints::ValueCommitV(point) => point,
            OrchardFixedPoints::ValueCommitR(point) => point,
            OrchardFixedPoints::CommitIvkBase(point) => point,
        }
    }
}

impl<C: CurveAffine> std::hash::Hash for OrchardFixedPoints<C> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match *self {
            OrchardFixedPoints::NullifierBase(_) => {
                state.write(&format!("{:?}", "NullifierBase").as_bytes())
            }
            OrchardFixedPoints::NoteCommitBase(_) => {
                state.write(&format!("{:?}", "NoteCommitBase").as_bytes())
            }
            OrchardFixedPoints::ValueCommitV(_) => {
                state.write(&format!("{:?}", "ValueCommitV").as_bytes())
            }
            OrchardFixedPoints::ValueCommitR(_) => {
                state.write(&format!("{:?}", "ValueCommitR").as_bytes())
            }
            OrchardFixedPoints::CommitIvkBase(_) => {
                state.write(&format!("{:?}", "CommitIvkBase").as_bytes())
            }
        }
    }
}

impl<C: CurveAffine> FixedPoints<C> for OrchardFixedPoints<C> {}

#[derive(Clone, Debug)]
pub(crate) struct CellValue<F: FieldExt> {
    cell: Cell,
    value: Option<F>,
}

impl<F: FieldExt> CellValue<F> {
    pub fn new(cell: Cell, value: Option<F>) -> Self {
        CellValue { cell, value }
    }
}

#[derive(Clone, Debug)]
pub struct EccScalarVar<C: CurveAffine> {
    value: Option<C::Scalar>,
    k_bits: Vec<CellValue<C::Base>>, // bitwise decomposition
}

#[derive(Clone, Debug)]
pub struct EccScalarFixed<C: CurveAffine> {
    value: Option<C::Scalar>,
    k_bits: Vec<CellValue<C::Base>>, // 3-bit decomposition
}

#[derive(Clone, Debug)]
pub struct EccPoint<F: FieldExt> {
    x: CellValue<F>,
    y: CellValue<F>,
}

#[derive(Clone, Debug)]
pub struct EccFixedPoint<C: CurveAffine> {
    fixed_point: OrchardFixedPoints<C>,
    lagrange_coeffs: Option<Vec<Vec<C::Base>>>,
    z: Option<Vec<C::Base>>,
}

impl<C: CurveAffine> EccInstructions<C> for EccChip<C> {
    type ScalarVar = EccScalarVar<C>;
    type ScalarFixed = EccScalarFixed<C>;
    type Point = EccPoint<C::Base>;
    type FixedPoint = EccFixedPoint<C>;
    type FixedPoints = OrchardFixedPoints<C>;

    /// Witnesses the given scalar as a private input to the circuit for variable-based scalar mul.
    fn witness_scalar_var(
        layouter: &mut impl Layouter<Self>,
        value: Option<C::Scalar>,
    ) -> Result<Self::ScalarVar, Error> {
        todo!()
    }

    /// Witnesses the given scalar as a private input to the circuit for fixed-based scalar mul.
    fn witness_scalar_fixed(
        layouter: &mut impl Layouter<Self>,
        value: Option<C::Scalar>,
    ) -> Result<Self::ScalarFixed, Error> {
        let config = layouter.config().clone();

        let scalar = layouter.assign_region(
            || "witness scalar for fixed-base mul",
            |mut region| witness_scalar_fixed::assign_region(value, &mut region, config.clone()),
        )?;

        Ok(scalar)
    }

    fn witness_point(
        layouter: &mut impl Layouter<Self>,
        value: Option<C::CurveExt>,
    ) -> Result<Self::Point, Error> {
        let config = layouter.config().clone();

        let point = layouter.assign_region(
            || "witness point",
            |mut region| witness_point::assign_region(value, &mut region, config.clone()),
        )?;

        Ok(point)
    }

    fn get_fixed(
        layouter: &mut impl Layouter<Self>,
        fixed_point: Self::FixedPoints,
    ) -> Result<Self::FixedPoint, Error> {
        let loaded = layouter.loaded();

        let lagrange_coeffs = loaded.get_lagrange_coeffs(fixed_point.clone());
        let z = loaded.get_z(fixed_point.clone());
        Ok(EccFixedPoint {
            fixed_point,
            lagrange_coeffs,
            z,
        })
    }

    fn add(
        layouter: &mut impl Layouter<Self>,
        a: &Self::Point,
        b: &Self::Point,
    ) -> Result<Self::Point, Error> {
        let config = layouter.config().clone();

        let point = layouter.assign_region(
            || "point addition",
            |mut region| add::assign_region(a, b, &mut region, config.clone()),
        )?;

        Ok(point)
    }

    fn add_complete(
        layouter: &mut impl Layouter<Self>,
        a: &Self::Point,
        b: &Self::Point,
    ) -> Result<Self::Point, Error> {
        todo!()
    }

    fn double(layouter: &mut impl Layouter<Self>, a: &Self::Point) -> Result<Self::Point, Error> {
        let config = layouter.config().clone();

        let point = layouter.assign_region(
            || "point doubling",
            |mut region| double::assign_region(a, &mut region, config.clone()),
        )?;

        Ok(point)
    }

    fn mul(
        layouter: &mut impl Layouter<Self>,
        scalar: &Self::ScalarVar,
        base: &Self::Point,
    ) -> Result<Self::Point, Error> {
        todo!()
    }

    fn mul_fixed(
        layouter: &mut impl Layouter<Self>,
        scalar: &Self::ScalarFixed,
        base: &Self::FixedPoint,
    ) -> Result<Self::Point, Error> {
        todo!()
    }
}
