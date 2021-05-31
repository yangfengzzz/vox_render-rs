/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use cgmath::*;
use crate::bounding_box::BoundingBox;

/**
 * A bounding sphere.
 * */
pub struct BoundingSphere {
    /** The center point of the sphere. */
    pub center: Vector3<f32>,
    /** The radius of the sphere. */
    pub radius: f32,
}

impl BoundingSphere {
    /**
     * Constructor of BoundingSphere.
     * @param center - The center point of the sphere
     * @param radius - The radius of the sphere
     */
    pub fn new(center: Option<Vector3<f32>>, radius: Option<f32>) -> BoundingSphere {
        return BoundingSphere {
            center: center.unwrap_or(Vector3::zero()),
            radius: radius.unwrap_or(0.0),
        };
    }
}

impl BoundingSphere {
    /**
     * Calculate a bounding sphere that fully contains the given points.
     * @param points - The given points
     * @param out - The calculated bounding sphere
     */
    pub fn from_points(points: Vec<Vector3<f32>>, out: &mut BoundingSphere) {
        if points.len() == 0 {
            panic!("points must be array and length must > 0");
        }

        let mut center = Vector3::zero();

        // Calculate the center of the sphere.
        for point in &points {
            center += *point;
        }

        // The center of the sphere.
        out.center = center / points.len() as f32;

        // Calculate the radius of the sphere.
        let mut radius = 0.0;
        for point in &points {
            let distance = Vector3::distance2(center, *point);
            if distance > radius {
                radius = distance;
            }
        }
        // The radius of the sphere.
        out.radius = f32::sqrt(radius);
    }

    /**
     * Calculate a bounding sphere from a given box.
     * @param box - The given box
     * @param out - The calculated bounding sphere
     */
    pub fn from_box(aabb: &BoundingBox, out: &mut BoundingSphere) {
        out.center.x = (aabb.min.x + aabb.max.x) * 0.5;
        out.center.y = (aabb.min.y + aabb.max.y) * 0.5;
        out.center.z = (aabb.min.z + aabb.max.z) * 0.5;
        out.radius = Vector3::distance(out.center, aabb.max.to_vec());
    }
}

impl Clone for BoundingSphere {
    fn clone(&self) -> Self {
        return Self {
            center: self.center,
            radius: self.radius,
        };
    }
}