# dropoint.py

import math
import numpy as np

import eve
from eve import intervention 


class Dropoint(intervention.MonoPlaneStatic):
    """
    Standalone intervention for DROPO-based Sim2Real transfer,
    replicating ArchVariety setup and adding sim2real hooks.
    """

    # number of domain parameters to optimise: [stiffness, friction, wallThickness]
    domain_params_dim = 3

    def __init__(
        self,
        episodes_between_arch_change: int = 1,
        stop_device_at_tree_end: bool = True,
        normalize_action: bool = False,
    ) -> None:
        # randomise aortic arch geometry every N episodes
        vessel_tree = eve.intervention.vesseltree.AorticArchRandom(
            episodes_between_change=episodes_between_arch_change,
            scale_diameter_array=[0.85],
            arch_types_filter=[eve.intervention.vesseltree.ArchType.I],
        )

        # guidewire device parameters
        device = eve.intervention.device.JShaped(
            name="guidewire",
            velocity_limit=(35, 3.14),
            length=450,
            tip_radius=12.1,
            tip_angle=0.4 * math.pi,
            tip_outer_diameter=0.7,
            tip_inner_diameter=0.0,
            straight_outer_diameter=0.89,
            straight_inner_diameter=0.0,
            poisson_ratio=0.49,
            young_modulus_tip=17e3,
            young_modulus_straight=80e3,
            mass_density_tip=0.000021,
            mass_density_straight=0.000021,
            visu_edges_per_mm=0.5,
            collis_edges_per_mm_tip=2,
            collis_edges_per_mm_straight=0.1,
            beams_per_mm_tip=1.4,
            beams_per_mm_straight=0.5,
            color=(0.0, 0.0, 0.0),
        )

        # simulation engine adapter
        simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.1)

        # fluoroscopy imaging
        fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
            simulation=simulation,
            vessel_tree=vessel_tree,
            image_frequency=7.5,
            image_rot_zx=[25, 0],
            image_center=[0, 0, 0],
            field_of_view=None,
        )

        # random target along vessel centerline
        target = eve.intervention.target.CenterlineRandom(
            vessel_tree=vessel_tree,
            fluoroscopy=fluoroscopy,
            threshold=5,
            branches=["lcca", "rcca", "lsa", "rsa", "bct", "co"],
        )

        super().__init__(
            vessel_tree,
            [device],
            simulation,
            fluoroscopy,
            target,
            stop_device_at_tree_end,
            normalize_action,
        )

    @property
    def episodes_between_arch_change(self) -> int:
        return self.vessel_tree.episodes_between_change
    
    def set_scene_inputs(self, insertion_point, insertion_direction, mesh_path):
        """
        Store scene initialisation values required for simulation.reset.
        Must be called before calling set_domain_params.
        """
        self._insertion_point = insertion_point
        self._insertion_direction = insertion_direction
        self._mesh_path = mesh_path

    def _get_root_node(self):
        sim = self.simulation
        print("[DEBUG] Starting sim:", sim)

        unwrap_limit = 5
        while unwrap_limit > 0 and hasattr(sim, "simulation"):
            print("[DEBUG] Unwrapping sim.simulation:", sim.simulation)
            sim = sim.simulation
            unwrap_limit -= 1

        # Ensure reset has been called to initialize .root
        if hasattr(sim, "reset") and getattr(sim, "root", None) is None:
            print("[DEBUG] Calling reset() to initialize SOFA scene")
            sim.reset(
                insertion_point=self._insertion_point,
                insertion_direction=self._insertion_direction,
                mesh_path=self._mesh_path,
                devices=self.devices,
            )

        if hasattr(sim, "root") and sim.root is not None:
            print("[DEBUG] Found root node in sim:", sim.root)
            return sim.root

        # fallback
        for attr in ("rootNode", "scene", "node", "_rootNode", "_root_node"):
            val = getattr(sim, attr, None)
            if val is not None:
                print(f"[DEBUG] Found attribute {attr}: {val}")
                return val

        raise AttributeError(
            f"[Dropoint] cannot find SOFA scene root on {sim!r}; "
            f"available attributes: {dir(sim)}"
        )

    def set_state_from_observation(self, observation):
        root = self._get_root_node()
        catheter_node = root.getChild("InstrumentCombined")
        if catheter_node:
            dofs_object = catheter_node.getObject("DOFs")
            if dofs_object:
                pos_data = dofs_object.findData("position")
                if pos_data:
                    positions = pos_data.value
                    new_positions = positions.copy()  # ensure writable
                    new_positions[0][:2] = observation[:2]
                    pos_data.value = new_positions
                    print("[Dropoint] Updated catheter tip position!")
                else:
                    raise ValueError("No 'position' data field found in DOFs object.")
            else:
                raise ValueError("DOFs object not found in InstrumentCombined.")


    def reset_to_state(self, observation: np.ndarray):
        """
        Alias for DROPO: DROPO expects `intervention.reset_to_state(obs)`.
        """
        return self.set_state_from_observation(observation)

    def set_domain_params(self, params: np.ndarray):
        """
        Apply domain randomisation parameters:
        params[0] → stiffness
        params[1] → friction
        params[2] → wallThickness
        """
        stiffness, friction, wallThk = float(params[0]), float(params[1]), float(params[2])
        root = self._get_root_node()

        # === Apply stiffness ===
        try:
            vessel = root.getChild("vesselTree")
            if vessel.findData("stiffness"):
                vessel.findData("stiffness").value = stiffness
                print(f"[✔] Applied stiffness → {stiffness}")
        except Exception as e:
            print(f"[✘] Could not apply stiffness: {e}")

        # === Apply friction (search through root objects safely) ===
        try:
            for obj in root.objects:
                try:
                    data = obj.findData("frictionCoef")
                    if data:
                        data.value = friction
                        print(f"[✔] Applied friction → {friction} on object: {getattr(obj, 'name', 'unknown')}")
                except Exception:
                    continue
        except Exception as e:
            print(f"[✘] Could not apply friction: {e}")

        # === Apply wall thickness ===
        try:
            wall_data = root.getChild("vesselTree").findData("thickness")
            if wall_data:
                wall_data.value = wallThk
                print(f"[✔] Applied wallThk → {wallThk}")
        except Exception as e:
            print(f"[✘] Could not apply wallThk: {e}")

        print(
            f"[Dropoint] domain params → stiffness={stiffness:.4f}, "
            f"friction={friction:.4f}, wallThk={wallThk:.4f}"
        )
