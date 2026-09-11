from infer_python import release_worker_hardware
from robot_handoff import disconnect_robot_hardware


class FakeCamera:
    def __init__(self):
        self.disconnects = 0

    def disconnect(self):
        self.disconnects += 1


class FakeBus:
    def __init__(self):
        self.disable_torque_values = []

    def disconnect(self, disable_torque=True):
        self.disable_torque_values.append(disable_torque)


class FakeRobot:
    def __init__(self):
        self.config = type("Config", (), {"disable_torque_on_disconnect": True})()
        self.cameras = {"top": FakeCamera(), "wrist": FakeCamera()}
        self.bus = FakeBus()
        self.disconnects = 0
        self.disconnect_torque_values = []

    def disconnect(self):
        self.disconnects += 1
        value = self.config.disable_torque_on_disconnect
        self.disconnect_torque_values.append(value)
        for camera in self.cameras.values():
            camera.disconnect()
        self.bus.disconnect(disable_torque=value)


def test_handoff_disconnect_releases_connections_without_disabling_torque():
    robot = FakeRobot()

    disconnect_robot_hardware(robot, keep_torque=True)

    assert [camera.disconnects for camera in robot.cameras.values()] == [1, 1]
    assert robot.bus.disable_torque_values == [False]
    assert robot.disconnects == 1
    assert robot.disconnect_torque_values == [False]
    assert robot.config.disable_torque_on_disconnect is True


def test_shutdown_disconnect_uses_normal_robot_cleanup():
    robot = FakeRobot()

    disconnect_robot_hardware(robot, keep_torque=False)

    assert robot.disconnects == 1
    assert robot.bus.disable_torque_values == [True]


def test_worker_handoff_keeps_model_runtime_but_releases_hardware_with_torque():
    robot = FakeRobot()
    runtime = {"robot": robot, "hardware_connected": True, "policy": object()}

    release_worker_hardware(runtime, keep_torque=True)

    assert runtime["robot"] is robot
    assert runtime["hardware_connected"] is False
    assert robot.bus.disable_torque_values == [False]
