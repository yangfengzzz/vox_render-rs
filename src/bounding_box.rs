/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use cgmath::*;
use crate::bounding_sphere::BoundingSphere;

/**
 * Axis Aligned Bound Box (AABB).
 */
pub struct BoundingBox {
    /** The minimum point of the box. */
    pub(crate) min: Point3<f32>,
    /** The maximum point of the box. */
    pub(crate) max: Point3<f32>,
}

impl BoundingBox {
    /**
     * Constructor of BoundingBox.
     * @param min - The minimum point of the box
     * @param max - The maximum point of the box
     */
    pub fn new(min: Option<Point3<f32>>, max: Option<Point3<f32>>) -> BoundingBox {
        return BoundingBox {
            min: min.unwrap_or(Point3::new(0.0, 0.0, 0.0)),
            max: max.unwrap_or(Point3::new(0.0, 0.0, 0.0)),
        };
    }
}

impl BoundingBox {
    /**
     * Get the center point of this bounding box.
     * @param out - The center point of this bounding box
     * @returns The center point of this bounding box
     */
    pub fn get_center(&self) -> Point3<f32> {
        let mut out = self.min.to_vec() + self.max.to_vec();
        out *= 0.5;
        return Point3::from_vec(out);
    }

    /**
     * Get the extent of this bounding box.
     * @param out - The extent of this bounding box
     * @returns The extent of this bounding box
     */
    pub fn get_extent(&self) -> Vector3<f32> {
        let mut out = self.max - self.min;
        out *= 0.5;
        return out;
    }

    /**
     * Get the eight corners of this bounding box.
     * @param out - An array of points representing the eight corners of this bounding box
     * @returns An array of points representing the eight corners of this bounding box
     */
    pub fn get_corners(&self) -> [Point3<f32>; 8] {
        let min_x = self.min.x;
        let min_y = self.min.y;
        let min_z = self.min.z;
        let max_x = self.max.x;
        let max_y = self.max.y;
        let max_z = self.max.z;

        let mut out = [Point3::new(0.0, 0.0, 0.0); 8];

        out[0] = Point3::new(min_x, max_y, max_z);
        out[1] = Point3::new(max_x, max_y, max_z);
        out[2] = Point3::new(max_x, min_y, max_z);
        out[3] = Point3::new(min_x, min_y, max_z);
        out[4] = Point3::new(min_x, max_y, min_z);
        out[5] = Point3::new(max_x, max_y, min_z);
        out[6] = Point3::new(max_x, min_y, min_z);
        out[7] = Point3::new(min_x, min_y, min_z);

        return out;
    }
}

impl BoundingBox {
    /**
     * Calculate a bounding box from the center point and the extent of the bounding box.
     * @param center - The center point
     * @param extent - The extent of the bounding box
     * @param out - The calculated bounding box
     */
    pub fn from_center_and_extent(center: &Point3<f32>, extent: &Vector3<f32>, out: &mut BoundingBox) {
        out.min = center - extent;
        out.max = center + extent;
    }

    /**
     * Calculate a bounding box that fully contains the given points.
     * @param points - The given points
     * @param out - The calculated bounding box
     */
    pub fn from_points(points: Vec<Point3<f32>>, out: &mut BoundingBox) {
        if points.len() == 0 {
            panic!("points must be array and length must > 0");
        }

        out.min.x = f32::MAX;
        out.min.y = f32::MAX;
        out.min.z = f32::MAX;
        out.max.x = -f32::MAX;
        out.max.y = -f32::MAX;
        out.max.z = -f32::MAX;

        for point in &points {
            crate::point3::min(*point, out.min, &mut out.min);
            crate::point3::max(*point, out.max, &mut out.max);
        }
    }

    /**
     * Calculate a bounding box from a given sphere.
     * @param sphere - The given sphere
     * @param out - The calculated bounding box
     */
    pub fn from_sphere(sphere: &BoundingSphere, out: &mut BoundingBox) {
        out.min.x = sphere.center.x - sphere.radius;
        out.min.y = sphere.center.y - sphere.radius;
        out.min.z = sphere.center.z - sphere.radius;
        out.max.x = sphere.center.x + sphere.radius;
        out.max.y = sphere.center.y + sphere.radius;
        out.max.z = sphere.center.z + sphere.radius;
    }

    /**
     * Transfrom a bounding box.
     * @param source - The original bounding box
     * @param matrix - The transform to apply to the bounding box
     * @param out - The transformed bounding box
     */
    pub fn transform(source: &BoundingBox, matrix: Matrix4<f32>, out: &mut BoundingBox) {
        // https://zeux.io/2010/10/17/aabb-from-obb-with-component-wise-abs/
        let mut center = source.get_center();
        let mut extent = source.get_extent();
        center = matrix.transform_point(center);

        extent.x = f32::abs(extent.x * matrix[0][0]) + f32::abs(extent.y * matrix[1][0]) + f32::abs(extent.z * matrix[2][0]);
        extent.y = f32::abs(extent.x * matrix[0][1]) + f32::abs(extent.y * matrix[1][1]) + f32::abs(extent.z * matrix[2][1]);
        extent.z = f32::abs(extent.x * matrix[0][2]) + f32::abs(extent.y * matrix[1][2]) + f32::abs(extent.z * matrix[2][2]);

        // set min、max
        out.min = center - extent;
        out.max = center + extent;
    }

    /**
     * Calculate a bounding box that is as large as the total combined area of the two specified boxes.
     * @param box1 - The first box to merge
     * @param box2 - The second box to merge
     * @param out - The merged bounding box
     * @returns The merged bounding box
     */
    pub fn merge(box1: &BoundingBox, box2: &BoundingBox) -> BoundingBox {
        let mut out = BoundingBox::new(None, None);
        crate::point3::min(box1.min, box2.min, &mut out.min);
        crate::point3::max(box1.max, box2.max, &mut out.max);
        return out;
    }
}

impl Clone for BoundingBox {
    fn clone(&self) -> Self {
        return Self {
            min: self.min,
            max: self.max,
        };
    }
}