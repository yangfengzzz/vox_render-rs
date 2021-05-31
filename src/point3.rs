/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use cgmath::Point3;

/**
 * Calculate a vector containing the smallest components of the specified vectors.
 * @param left - The first vector
 * @param right - The second vector
 * @param out - The vector containing the smallest components of the specified vectors
 */
pub fn min(left: Point3<f32>, right: Point3<f32>, out: &mut Point3<f32>) {
    out.x = f32::min(left.x, right.x);
    out.y = f32::min(left.y, right.y);
    out.z = f32::min(left.z, right.z);
}

/**
 * Calculate a vector containing the largest components of the specified vectors.
 * @param left - The first vector
 * @param right - The second vector
 * @param out - The vector containing the largest components of the specified vectors
 */
pub fn max(left: Point3<f32>, right: Point3<f32>, out: &mut Point3<f32>) {
    out.x = f32::max(left.x, right.x);
    out.y = f32::max(left.y, right.y);
    out.z = f32::max(left.z, right.z);
}