use std::marker::PhantomData;

use super::SinsemillaInstructions;
use halo2::{
    arithmetic::{CurveAffine, CurveExt, Field, FieldExt},
    circuit::{Chip, Layouter},
    plonk::{Advice, Column, ConstraintSystem, Error, Expression, Fixed},
    poly::Rotation,
};

use group::Curve;

mod generator_table;
use generator_table::*;

const GROUP_HASH_Q: &str = "z.cash:SinsemillaQ";
const GROUP_HASH_S: &str = "z.cash:SinsemillaS";

const K: usize = 10;
const C: usize = 253;

fn lebs2ip_32(bits: &[bool]) -> u32 {
    bits.iter()
        .enumerate()
        .fold(0u32, |acc, (i, b)| acc + if *b { 1 << i } else { 0 })
}

/// Configuration for the Sinsemilla chip
#[derive(Clone, Debug)]
pub struct SinsemillaConfig {
    columns: SinsemillaColumns,
    lookup_table: GeneratorTable,
}

/// Columns needed for one Sinsemilla hash
#[derive(Clone, Debug)]
pub struct SinsemillaColumns {
    sinsemilla: Column<Fixed>,
    x_a: Column<Advice>,
    z: Column<Advice>,
    lambda1: Column<Advice>,
    lambda2: Column<Advice>,
    x_p: Column<Advice>,
}

impl SinsemillaColumns {
    /// Construct a new instance of `SinsemillaColumns`
    pub fn new(
        sinsemilla: Column<Fixed>,
        x_a: Column<Advice>,
        z: Column<Advice>,
        lambda1: Column<Advice>,
        lambda2: Column<Advice>,
        x_p: Column<Advice>,
    ) -> Self {
        SinsemillaColumns {
            sinsemilla,
            x_a,
            z,
            lambda1,
            lambda2,
            x_p,
        }
    }
}

/// A chip implementing SinsemillaInstructions
#[derive(Debug)]
pub struct SinsemillaChip<C: CurveAffine> {
    _marker_c: PhantomData<C>,
}

impl<F: FieldExt, C: CurveAffine<Base = F>> SinsemillaChip<C> {
    /// Configures this chip for use in a circuit.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        columns: SinsemillaColumns,
    ) -> SinsemillaConfig {
        // Fixed column for Sinsemilla selector
        let sinsemilla_cur = meta.query_fixed(columns.sinsemilla, Rotation::cur());

        // m_i = z_{i + 1} - (z_i * 2^k)
        let z_cur = meta.query_advice(columns.z, Rotation::cur());
        let z_next = meta.query_advice(columns.z, Rotation::next());
        let m = z_next - z_cur * F::from_u64((1 << K) as u64);

        // y_a = (1/2) ⋅ (lambda1 + lambda2) ⋅ (x_a - (lambda1^2 - x_a - x_p))
        let lambda1_cur = meta.query_advice(columns.lambda1, Rotation::cur());
        let lambda2_cur = meta.query_advice(columns.lambda2, Rotation::cur());
        let x_a_cur = meta.query_advice(columns.x_a, Rotation::cur());
        let x_p_cur = meta.query_advice(columns.x_p, Rotation::cur());
        let y_a_cur = (lambda1_cur.clone() + lambda2_cur.clone())
            * (x_a_cur.clone()
                - (lambda1_cur.clone() * lambda1_cur.clone() - x_a_cur.clone() - x_p_cur.clone()))
            * F::TWO_INV;

        // y_p = y_a - lambda1 ⋅ (x_a - x_p)
        let y_p = y_a_cur.clone() - lambda1_cur.clone() * (x_a_cur.clone() - x_p_cur.clone());

        let (x_p_init, y_p_init) = get_s_by_idx::<C>(0).to_affine().get_xy().unwrap();

        let lookup_table = GeneratorTable::configure::<C>(
            meta,
            sinsemilla_cur.clone() * m
                + (Expression::Constant(F::one()) - sinsemilla_cur.clone()) * F::zero(),
            sinsemilla_cur.clone() * x_p_cur.clone()
                + (Expression::Constant(F::one()) - sinsemilla_cur.clone()) * x_p_init,
            sinsemilla_cur.clone() * y_p
                + (Expression::Constant(F::one()) - sinsemilla_cur.clone()) * y_p_init,
        );

        let lambda1_next = meta.query_advice(columns.lambda1, Rotation::next());
        let lambda2_next = meta.query_advice(columns.lambda2, Rotation::next());
        let x_a_next = meta.query_advice(columns.x_a, Rotation::next());
        let x_p_next = meta.query_advice(columns.x_p, Rotation::next());
        let y_a_next = (lambda1_next.clone() + lambda2_next)
            * (x_a_next.clone()
                - (lambda1_next.clone() * lambda1_next - x_a_next.clone() - x_p_next))
            * F::TWO_INV;

        // Sinsemilla expr1 gate
        meta.create_gate("Sinsemilla expr1", |_| {
            // λ_{2,i}^2 − x_{A,i+1} −(λ_{1,i}^2 − x_{A,i} − x_{P,i}) − x_{A,i} = 0
            let expr1 = lambda2_cur.clone() * lambda2_cur.clone()
                - x_a_next.clone()
                - (lambda1_cur.clone() * lambda1_cur)
                + x_p_cur;

            sinsemilla_cur.clone() * expr1
        });

        // Sinsemilla expr2 gate
        meta.create_gate("Sinsemilla expr2", |_| {
            // λ_{2,i}⋅(x_{A,i} − x_{A,i+1}) − y_{A,i} − y_{A,i+1} = 0
            let expr2 = lambda2_cur * (x_a_cur - x_a_next) - y_a_cur - y_a_next;

            sinsemilla_cur.clone() * expr2
        });

        SinsemillaConfig {
            columns,
            lookup_table,
        }
    }
}

impl<C: CurveAffine> Chip for SinsemillaChip<C> {
    type Config = SinsemillaConfig;
    type Field = C::Base;
    type Loaded = ();

    fn load(layouter: &mut impl Layouter<Self>) -> Result<Self::Loaded, Error> {
        let table = layouter.config().lookup_table.clone();
        table.load(layouter)
    }
}

impl<C: CurveAffine> SinsemillaInstructions<C> for SinsemillaChip<C> {
    type Message = Vec<bool>;

    fn extract(point: &C::Curve) -> C::Base {
        todo!()
    }

    fn Q(domain_prefix: &str) -> C::CurveExt {
        C::CurveExt::hash_to_curve(GROUP_HASH_Q)(domain_prefix.as_bytes())
    }

    fn hash_to_point(
        layouter: &mut impl Layouter<Self>,
        domain_prefix: &str,
        message: Self::Message,
    ) -> Result<C, Error> {
        let config = layouter.config().clone();

        // Pad message to nearest multiple of K bits
        assert!(message.len() <= K * C);
        let pad = message.len() % K;
        let padded: Vec<_> = message
            .into_iter()
            .chain(std::iter::repeat(false).take(pad))
            .collect();

        // Parse message into `K`-bit words
        let words: Vec<u32> = padded.chunks(K).map(|chunk| lebs2ip_32(chunk)).collect();

        // Get (x_p, y_p) for each word. We precompute this here so that we can use `batch_normalize()`.
        let generators_projective: Vec<_> =
            words.iter().map(|word| get_s_by_idx::<C>(*word)).collect();
        let mut generators = vec![C::default(); generators_projective.len()];
        C::Curve::batch_normalize(&generators_projective, &mut generators);
        let generators: Vec<(C::Base, C::Base)> =
            generators.iter().map(|gen| gen.get_xy().unwrap()).collect();

        // Initialize `(x_a, y_a)` to be `(x_q, y_q)`
        let q = Self::Q(domain_prefix);
        let (mut x_a, mut y_a) = q.to_affine().get_xy().unwrap();

        layouter.assign_region(
            || "Assign message",
            |mut region| {
                // Initialize `(x_a, y_a)` to be `(x_q, y_q)`
                let q = Self::Q(domain_prefix).to_affine().get_xy().unwrap();
                x_a = q.0;
                y_a = q.1;

                // Initialize `z_0` = 0;
                let mut z = 0u64;

                if words.len() > 0 {
                    for row in 0..(words.len() - 1) {
                        // Activate `Sinsemilla` custom gate
                        region.assign_fixed(
                            || "Sinsemilla expr1",
                            config.columns.sinsemilla,
                            row,
                            || Ok(C::Base::one()),
                        )?;
                    }
                }

                // Assign initialized values
                region.assign_advice(|| "z_0", config.columns.z, 0, || Ok(C::Base::from_u64(z)))?;
                region.assign_advice(|| "x_q", config.columns.x_a, 0, || Ok(x_a))?;

                for row in 0..words.len() {
                    let word = words[row];
                    let gen = generators[row];
                    let x_p = gen.0;
                    let y_p = gen.1;

                    // Assign `x_p`
                    region.assign_advice(|| "x_p", config.columns.x_p, row, || Ok(x_p))?;

                    // Compute and assign `z` for the next row
                    z = z * (1 << K) + (word as u64);
                    region.assign_advice(
                        || "z",
                        config.columns.z,
                        row + 1,
                        || Ok(C::Base::from_u64(z)),
                    )?;

                    // Compute and assign `lambda1, lambda2`
                    let lambda1 = (y_a - y_p) * (x_a - x_p).invert().unwrap();
                    let x_r = lambda1 * lambda1 - x_a - x_p;
                    let lambda2 =
                        C::Base::from_u64(2) * y_a * (x_a - x_r).invert().unwrap() - lambda1;
                    region.assign_advice(
                        || "lambda1",
                        config.columns.lambda1,
                        row,
                        || Ok(lambda1),
                    )?;
                    region.assign_advice(
                        || "lambda2",
                        config.columns.lambda2,
                        row,
                        || Ok(lambda2),
                    )?;

                    // Compute and assign `x_a` for the next row
                    let x_a_new = lambda2 * lambda2 - x_a - x_r;
                    y_a = lambda2 * (x_a - x_a_new) - y_a;
                    x_a = x_a_new;
                    region.assign_advice(|| "x_a", config.columns.x_a, row + 1, || Ok(x_a))?;
                }

                Ok(())
            },
        )?;

        Ok(C::from_xy(x_a, y_a).unwrap())
    }

    fn hash(
        layouter: &mut impl Layouter<Self>,
        domain_prefix: &str,
        message: Self::Message,
    ) -> Result<C::Base, Error> {
        todo!()
    }

    fn commit(
        domain_prefix: &str,
        msg: Self::Message,
        r: &C::Scalar,
    ) -> Result<C::CurveExt, Error> {
        todo!()
    }

    fn short_commit(
        domain_prefix: &str,
        msg: Self::Message,
        r: &C::Scalar,
    ) -> Result<C::Base, Error> {
        todo!()
    }
}
