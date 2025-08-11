import eve
from eve import visualisation
from eve import env

class BenchEnv(env.Env):
    """
    Benchmark environment based on eve.Env
    Wraps a SimulatedIntervention for wither training or evaluation
    """
    def __init__(
        self,
        intervention: eve.intervention.SimulatedIntervention,
        mode: str = "train",
        visualisation: bool = False,
        n_max_steps=1000,
    ) -> None:
        """
        Initialise the BenchEnv
        Parameters:
        - intervention: The simulated scenario (SimulatedIntervention)
        - mode: "train" or "eval" - this adjusts tuncation criteria
        - visualisation: whether to enable real-time visualisation (does this mean the simulated visualisation or the camera tracking? need to ask)
        - n_max_steps: Maximum number of steps per episode
        """
        self.mode = mode
        self.visualisation = visualisation
        # Starting point for episode (initial catheter position)
        start = eve.start.InsertionPoint(intervention)
        # Path planning / evaluation for the catheter trajectory
        pathfinder = eve.pathfinder.BruteForceBFS(intervention=intervention)
        
        # Observation

        # 1) Tracking catheter tip in 2D
        tracking = eve.observation.Tracking2D(intervention, n_points=3, resolution=2)
        tracking = eve.observation.wrapper.NormalizeTracking2DEpisode(
            tracking, intervention
        )
        tracking = eve.observation.wrapper.Memory(
            tracking, 2, eve.observation.wrapper.MemoryResetMode.FILL
        )
        # 2) Target position (normalised)
        target_state = eve.observation.Target2D(intervention)
        target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
            target_state, intervention
        )
        # 3) Last action taken (normalised)
        last_action = eve.observation.LastAction(intervention)
        last_action = eve.observation.wrapper.Normalize(last_action)

        # Combine observations into a dictionary 
        observation = eve.observation.ObsDict(
            {
                "tracking": tracking,
                "target": target_state,
                "last_action": last_action,
            }
        )

        # Reward
        target_reward = eve.reward.TargetReached(
            intervention,
            factor=1.0,
            final_only_after_all_interim=False,
        )
        step_reward = eve.reward.Step(factor=-0.005)  #Small step to encourage efficiency
        path_delta = eve.reward.PathLengthDelta(pathfinder, 0.001)  #Reward progress toward target
        reward = eve.reward.Combination([target_reward, path_delta, step_reward])

        # Terminal and Truncation
        terminal = eve.terminal.TargetReached(intervention) #Done when target reached

        max_steps = eve.truncation.MaxSteps(n_max_steps) #Episode ends after max steps
        vessel_end = eve.truncation.VesselEnd(intervention)  #Episode ends if vessel end reached
        sim_error = eve.truncation.SimError(intervention)  #Episode ends if sim error occurs

        if mode == "train":
            #During training: combine max steps, vessel end, and sim errors
            truncation = eve.truncation.Combination([max_steps, vessel_end, sim_error])
        else:
            #Evaluatio nmode: only use max steps as stopping condition
            truncation = max_steps

        # Info/Metrics to log
        target_reached = eve.info.TargetReached(intervention, name="success")
        path_ratio = eve.info.PathRatio(pathfinder)
        steps = eve.info.Steps()
        trans_speed = eve.info.AverageTranslationSpeed(intervention)
        trajectory_length = eve.info.TrajectoryLength(intervention)
        info = eve.info.Combination(
            [target_reached, path_ratio, steps, trans_speed, trajectory_length]
        )

        #Visualisation (if enabled)
        if visualisation:
            #in visualisation mode, disable multi-process for smoother rendering
            intervention.make_non_mp()
            visu = eve.visualisation.SofaPygame(intervention)
        else:
            #enables multi-process for faster data collection
            intervention.make_mp()
            visu = None
        
        # Initialise parent class (eve.Env)
        super().__init__(
            intervention,
            observation,
            reward,
            terminal,
            truncation=truncation,
            start=start,
            pathfinder=pathfinder,
            visualisation=visu,
            info=info,
            interim_target=None,
        )
        print("BenchEnv observation space:", self.observation.space)

