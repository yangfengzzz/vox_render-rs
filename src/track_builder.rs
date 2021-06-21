/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::raw_track::*;
use crate::track::*;

// Defines the class responsible of building runtime track instances from
// offline tracks.The input raw track is first validated. Runtime conversion of
// a validated raw track cannot fail. Note that no optimization is performed on
// the data at all.
pub struct TrackBuilder {}

impl TrackBuilder {
    // Creates a Track based on _raw_track and *this builder parameters.
    // Returns a track instance on success, an empty unique_ptr on failure. See
    // Raw*Track::Validate() for more details about failure reasons.
    // The track is returned as an unique_ptr as ownership is given back to the
    // caller.
    pub fn apply_float(_input: &RawFloatTrack) -> FloatTrack {
        todo!()
    }
    pub fn apply_float2(_input: &RawFloat2Track) -> Float2Track {
        todo!()
    }
    pub fn apply_float3(_input: &RawFloat3Track) -> Float3Track {
        todo!()
    }
    pub fn apply_float4(_input: &RawFloat4Track) -> Float4Track {
        todo!()
    }
    pub fn apply_quaternion(_input: &RawQuaternionTrack) -> QuaternionTrack {
        todo!()
    }

    fn build<_RawTrack, _Track>(_input: &_RawTrack) -> _Track {
        todo!()
    }
}