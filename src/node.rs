/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use std::cell::RefCell;
use std::rc::Rc;
use cgmath::*;
use crate::aabb::AABB;

#[derive(Copy, Clone)]
pub struct Node {
    name: Option<String>,
    position: Vector3<f32>,
    rotation: Vector3<f32>,
    scale: Vector3<f32>,

    bounding_box: AABB,

    parent: Option<Rc<RefCell<Node>>>,
    children: Vec<Rc<RefCell<Node>>>,
}

impl Node {
    pub fn new() -> Node {
        return Node {
            name: None,
            position: Vector3::zero(),
            rotation: Vector3::zero(),
            scale: Vector3::new(1.0, 1.0, 1.0),
            bounding_box: AABB::zero(),
            parent: None,
            children: vec![],
        };
    }
}

impl Node {
    pub fn quaternion(&self) -> Quaternion<f32> {
        let rotation_x = Matrix3::from_angle_x(Deg(self.rotation.x));
        let rotation_y = Matrix3::from_angle_y(Deg(self.rotation.y));
        let rotation_z = Matrix3::from_angle_z(Deg(self.rotation.z));
        return Quaternion::from(rotation_x * rotation_y * rotation_z);
    }

    pub fn model_matrix(&self) -> Matrix4<f32> {
        let translate_matrix = Matrix4::from_translation(self.position);
        let rotate_matrix = Matrix4::from(self.quaternion());
        let scale_matrix = Matrix4::from_nonuniform_scale(self.scale.x, self.scale.y, self.scale.z);
        return translate_matrix * rotate_matrix * scale_matrix;
    }

    pub fn world_transform(&self) -> Matrix4<f32> {
        if let Some(parent) = &self.parent {
            return parent.borrow().world_transform() * self.model_matrix();
        }
        return self.model_matrix();
    }

    pub fn forward_vector(&self) -> Vector3<f32> {
        return Vector3::new(f32::sin(self.rotation.y), 0.0, f32::cos(self.rotation.y)).normalize();
    }

    pub fn right_vector(&self) -> Vector3<f32> {
        return Vector3::new(self.forward_vector().z, self.forward_vector().y, -self.forward_vector().x);
    }

    pub fn size(&self) -> Vector3<f32> {
        return self.bounding_box.max_bounds - self.bounding_box.min_bounds;
    }
}

impl Node {
    pub fn add(&mut self, child_node: Rc<RefCell<Node>>) {
        self.children.push(child_node);
        child_node.borrow_mut().parent = Some(Rc::new(RefCell::new(*self)));
    }

    pub fn remove(&mut self, child_node: Rc<RefCell<Node>>) {
        for child in &mut child_node.borrow().children {
            child.borrow_mut().parent = self;
            self.children.push(child.clone());
        }
        child_node.borrow_mut().children.clear();
    }
}