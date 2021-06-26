/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::simd_math::Float4x4;

// Provides per-vertex matrix palette skinning job implementation.
// Skinning is the process of creating the association of skeleton joints with
// some vertices of a mesh. Portions of the mesh's skin can normally be
// associated with multiple joints, each one having a weight. The sum of the
// weights for a vertex is equal to 1. To calculate the final position of the
// vertex, each joint transformation is applied to the vertex position, scaled
// by its corresponding weight. This algorithm is called matrix palette skinning
// because of the set of joint transformations (stored as transform matrices)
// form a palette for the skin vertex to choose from.
//
// This job iterates and transforms vertices (points and vectors) provided as
// input using the matrix palette skinning algorithm. The implementation
// supports any number of joints influences per vertex, and can transform one
// point (vertex position) and two vectors (vertex normal and tangent) per loop
// (aka vertex). It assumes bi-normals aren't needed as they can be rebuilt
// from the normal and tangent with a lower cost than skinning (a single cross
// product).
// Input and output buffers must be provided with a stride value (aka the
// number of bytes from a vertex to the next). This allows the job to support
// vertices packed as array-of-structs (array of vertices with positions,
// normals...) or struct-of-arrays (buffers of points, buffer of normals...).
// The skinning job optimizes every code path at maximum. The loop is indeed not
// the same depending on the number of joints influencing a vertex (or if there
// are normals to transform). To maximize performances, application should
// partition its vertices based on their number of joints influences, and call
// a different job for every vertices set.
// Joint matrices are accessed using the per-vertex joints indices provided as
// input. These matrices must be pre-multiplied with the inverse of the skeleton
// bind-pose matrices. This allows to transform vertices to joints local space.
// In case of non-uniform-scale matrices, the job proposes to transform vectors
// using an optional set of matrices, whose are usually inverse transpose of
// joints matrices (see http://www.glprogramming.com/red/appendixf.html). This
// code path is less efficient than the one without this matrices set, and
// should only be used when input matrices have non uniform scaling or shearing.
// The job does not owned the buffers (in/output) and will thus not delete them
// during job's destruction.
pub struct SkinningJob<'a> {
    // Number of vertices to transform. All input and output arrays must store at
    // least this number of vertices.
    pub vertex_count: i32,

    // Maximum number of joints influencing each vertex. Must be greater than 0.
    // The number of influences drives how joint_indices and joint_weights are
    // sampled:
    // - influences_count joint indices are red from joint_indices for each
    // vertex.
    // - influences_count - 1 joint weights are red from joint_weightrs for each
    // vertex. The weight of the last joint is restored (weights are normalized).
    pub influences_count: i32,

    // Array of matrices for each joint. Joint are indexed through indices array.
    pub joint_matrices: &'a [Float4x4],

    // Optional array of inverse transposed matrices for each joint. If provided,
    // this array is used to transform vectors (normals and tangents), otherwise
    // joint_matrices array is used.
    // As explained here (http://www.glprogramming.com/red/appendixf.html) in the,
    // red book, transforming normals requires a special attention when the
    // transformation matrix has scaling or shearing. In this case the right
    // transformation is done by the inverse transpose of the transformation that
    // transforms points. Any rotation matrix is good though.
    // These matrices are optional as they might by costly to compute, and also
    // fall into a more costly code path in the skinning algorithm.
    pub joint_inverse_transpose_matrices: &'a [Float4x4],

    // Array of joints indices. This array is used to indexes matrices in joints
    // array.
    // Each vertex has influences_max number of indices, meaning that the size of
    // this array must be at least influences_max * vertex_count.
    pub joint_indices: &'a [u16],
    pub joint_indices_stride: usize,

    // Array of joints weights. This array is used to associate a weight to every
    // joint that influences a vertex. The number of weights required per vertex
    // is "influences_max - 1". The weight for the last joint (for each vertex) is
    // restored at runtime thanks to the fact that the sum of the weights for each
    // vertex is 1.
    // Each vertex has (influences_max - 1) number of weights, meaning that the
    // size of this array must be at least (influences_max - 1)* vertex_count.
    pub joint_weights: &'a [f32],
    pub joint_weights_stride: usize,

    // Input vertex positions array (3 float values per vertex) and stride (number
    // of bytes between each position).
    // Array length must be at least vertex_count * in_positions_stride.
    pub in_positions: &'a [f32],
    pub in_positions_stride: usize,

    // Input vertex normals (3 float values per vertex) array and stride (number
    // of bytes between each normal).
    // Array length must be at least vertex_count * in_normals_stride.
    pub in_normals: &'a [f32],
    pub in_normals_stride: usize,

    // Input vertex tangents (3 float values per vertex) array and stride (number
    // of bytes between each tangent).
    // Array length must be at least vertex_count * in_tangents_stride.
    pub in_tangents: &'a [f32],
    pub in_tangents_stride: usize,

    // Output vertex positions (3 float values per vertex) array and stride
    // (number of bytes between each position).
    // Array length must be at least vertex_count * out_positions_stride.
    pub out_positions: &'a mut [f32],
    pub out_positions_stride: usize,

    // Output vertex normals (3 float values per vertex) array and stride (number
    // of bytes between each normal).
    // Note that output normals are not normalized by the skinning job. This task
    // should be handled by the application, who knows if transform matrices have
    // uniform scale, and if normals are re-normalized later in the rendering
    // pipeline (shader vertex transformation stage).
    // Array length must be at least vertex_count * out_normals_stride.
    pub out_normals: &'a mut [f32],
    pub out_normals_stride: usize,

    // Output vertex positions (3 float values per vertex) array and stride
    // (number of bytes between each tangent).
    // Like normals, Note that output tangents are not normalized by the skinning
    // job.
    // Array length must be at least vertex_count * out_tangents_stride.
    pub out_tangents: &'a mut [f32],
    pub out_tangents_stride: usize,
}

impl<'a> SkinningJob<'a> {
    // Default constructor, initializes default values.
    pub fn new() -> SkinningJob<'a> {
        return SkinningJob {
            vertex_count: 0,
            influences_count: 0,
            joint_matrices: &[],
            joint_inverse_transpose_matrices: &[],
            joint_indices: &[],
            joint_indices_stride: 0,
            joint_weights: &[],
            joint_weights_stride: 0,
            in_positions: &[],
            in_positions_stride: 0,
            in_normals: &[],
            in_normals_stride: 0,
            in_tangents: &[],
            in_tangents_stride: 0,
            out_positions: &mut [],
            out_positions_stride: 0,
            out_normals: &mut [],
            out_normals_stride: 0,
            out_tangents: &mut [],
            out_tangents_stride: 0,
        };
    }

    // Validates job parameters.
    // Returns true for a valid job, false otherwise:
    // - if any range is invalid. See each range description.
    // - if normals are provided but positions aren't.
    // - if tangents are provided but normals aren't.
    // - if no output is provided while an input is. For example, if input normals
    // are provided, then output normals must also.
    pub fn validate(&self) -> bool {
        // Start validation of all parameters.
        let mut valid = true;

        // Checks influences bounds.
        valid &= self.influences_count > 0;

        // Checks joints matrices, required.
        valid &= !self.joint_matrices.is_empty();

        // Prepares local variables used to compute buffer size.
        let vertex_count_minus_1 = match self.vertex_count > 0 {
            true => self.vertex_count - 1,
            false => 0
        } as usize;
        let vertex_count_at_least_1 = (self.vertex_count > 0) as usize;

        // Checks indices, required.
        valid &= self.joint_indices.len() >=
            self.joint_indices_stride * vertex_count_minus_1 + self.influences_count as usize * vertex_count_at_least_1;

        // Checks weights, required if influences_count > 1.
        if self.influences_count != 1 {
            valid &=
                self.joint_weights.len() >=
                    self.joint_weights_stride * vertex_count_minus_1 + (self.influences_count - 1) as usize * vertex_count_at_least_1;
        }

        // Checks positions, mandatory.
        valid &= self.in_positions.len() >=
            self.in_positions_stride * vertex_count_minus_1 + 3 * vertex_count_at_least_1;
        valid &= !self.out_positions.is_empty();
        valid &= self.out_positions.len() >=
            self.out_positions_stride * vertex_count_minus_1 + 3 * vertex_count_at_least_1;

        // Checks normals, optional.
        if !self.in_normals.is_empty() {
            valid &= self.in_normals.len() >=
                self.in_normals_stride * vertex_count_minus_1 + 3 * vertex_count_at_least_1;
            valid &= !self.out_normals.is_empty();
            valid &= self.out_normals.len() >=
                self.out_normals_stride * vertex_count_minus_1 + 3 * vertex_count_at_least_1;

            // Checks tangents, optional but requires normals.
            if !self.in_tangents.is_empty() {
                valid &= self.in_tangents.len() >=
                    self.in_tangents_stride * vertex_count_minus_1 + 3 * vertex_count_at_least_1;
                valid &= !self.out_tangents.is_empty();
                valid &= self.out_tangents.len() >=
                    self.out_tangents_stride * vertex_count_minus_1 + 3 * vertex_count_at_least_1;
            }
        } else {
            // Tangents are not supported if normals are not there.
            valid &= self.in_tangents.is_empty();
        }

        return valid;
    }

    // Runs job's skinning task.
    // The job is validated before any operation is performed, see validate() for
    // more details.
    // Returns false if *this job is not valid.
    pub fn run(&mut self) -> bool {
        todo!()
    }
}