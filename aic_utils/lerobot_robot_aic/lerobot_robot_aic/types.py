from typing import TypedDict

MotionUpdateActionDict = TypedDict(
    "MotionUpdateActionDict",
    {
        "linear.x": float,
        "linear.y": float,
        "linear.z": float,
        "angular.x": float,
        "angular.y": float,
        "angular.z": float,
    },
)

PoseTargetActionDict = TypedDict(
    "PoseTargetActionDict",
    {
        "position.x": float,
        "position.y": float,
        "position.z": float,
        "orientation.w": float,
        "orientation.x": float,
        "orientation.y": float,
        "orientation.z": float,
    },
)

JointMotionUpdateActionDict = TypedDict(
    "JointMotionUpdateActionDict",
    {
        "shoulder_pan_joint": float,
        "shoulder_lift_joint": float,
        "elbow_joint": float,
        "wrist_1_joint": float,
        "wrist_2_joint": float,
        "wrist_3_joint": float,
    },
)
