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
pub(crate) fn decimate<Key: Clone, _Adapter: DecimateType<Key>>(_src: &Vec<Key>, _adapter: &_Adapter,
                                                                _tolerance: f32, _dest: &mut Vec<Key>) {
    // Early out if not enough data.
    if _src.len() < 2 {
        *_dest = (*_src).clone();
        return;
    }

    // Stack of segments to process.
    let mut segments: Vec<(usize, usize)> = Vec::new();

    // Bit vector of all points to included.
    let mut included: Vec<bool> = Vec::new();
    included.resize(_src.len(), false);

    // Pushes segment made from first and last points.
    segments.push((0, _src.len() - 1));
    included[0] = true;
    included[_src.len() - 1] = true;

    // Empties segments stack.
    while !segments.is_empty() {
        // Pops next segment to process.
        let segment = *segments.last().unwrap();
        segments.pop();

        // Looks for the furthest point from the segment.
        let mut max = -1.0;
        let mut candidate = segment.0;
        let left = &_src[segment.0];
        let right = &_src[segment.1];
        for i in (segment.0 + 1)..(segment.1) {
            debug_assert!(!included[i] && "Included points should be processed once only.".parse().unwrap_or(true));
            let test = &_src[i];
            if !_adapter.decimable(test) {
                candidate = i;
                break;
            } else {
                let distance = _adapter.distance(&_adapter.lerp(left, right, test), test);
                if distance > _tolerance && distance > max {
                    max = distance;
                    candidate = i;
                }
            }
        }

        // If found, include the point and pushes the 2 new segments (before and
        // after the new point).
        if candidate != segment.0 {
            included[candidate] = true;
            if candidate - segment.0 > 1 {
                segments.push((segment.0, candidate));
            }
            if segment.1 - candidate > 1 {
                segments.push((candidate, segment.1));
            }
        }
    }

    // Copy all included points.
    _dest.clear();
    for i in 0.._src.len() {
        if included[i] {
            _dest.push(_src[i].clone());
        }
    }

    // Removes last key if constant.
    if _dest.len() > 1 {
        let last = _dest.last().unwrap();
        let penultimate = &_dest[_dest.len() - 2];
        let distance = _adapter.distance(penultimate, last);
        if _adapter.decimable(last) && distance <= _tolerance {
            _dest.pop();
        }
    }
}