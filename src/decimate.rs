/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::animation_optimizer::DecimateType;

// Decimation algorithm based on Ramer–Douglas–Peucker.
// https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
// _Track must have std::vector interface.
// Adapter must have the following interface:
// struct Adapter {
//  bool decimable(const Key&) const;
//  Key lerp(const Key& _left, const Key& _right, const Key& _ref) const;
//  float distance(const Key& _a, const Key& _b) const;
// };
pub(crate) fn decimate<Key, _Adapter: DecimateType<Key>>(_src: &Vec<Key>, _adapter: &_Adapter,
                                                         _tolerance: f32, _dest: &mut Vec<Key>) {
    todo!()
}