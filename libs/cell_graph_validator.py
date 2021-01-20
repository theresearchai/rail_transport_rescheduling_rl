from libs.cell_graph_dispatcher import CellGraphDispatcher
from libs.dummy_observation import DummyObservationBuilder

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.envs.rail_env import RailAgentStatus

import numpy as np
import time


class CellGraphValidator:

    @staticmethod
    def multiple_tests(dispatcher_function,
                       width, height, trains, seed,
                       cities=None, rails_between_cities=None, rails_in_city=None,
                       malfunction_rate=None, prop_malfunction=None, min_prop=None, max_prop=None,
                       show_map=False):

        np.random.seed(seed)
        N = len(width)
        if cities is None:
            cities = np.random.randint(2, 35, (N,))
        if rails_between_cities is None:
            rails_between_cities = np.random.randint(2, 4, (N,))
        if rails_in_city is None:
            rails_in_city = np.random.randint(3, 6, (N,))
        if malfunction_rate is None:
            malfunction_rate = np.random.randint(500, 4000, (N,))
        if prop_malfunction is None:
            prop_malfunction = np.random.uniform(0.01, 0.01, (N,))
        if min_prop is None:
            min_prop = np.random.randint(20, 80, (N,))
        if max_prop is None:
            max_prop = np.random.randint(20, 80, (N,))
            max_prop = np.maximum(max_prop, min_prop)

        res_finished = []
        res_times = []

        for i in range(N):
            print("="*15)
            print("Test ", i)
            print("="*15)

            d = CellGraphValidator.single_test(dispatcher_function, width[i], height[i], trains[i], i, cities[i],
                                               rails_between_cities[i], rails_in_city[i],
                                               malfunction_rate[i], prop_malfunction[i],
                                               min_prop[i], max_prop[i],
                                               show_map=show_map)
            res_finished.append(d["finished"])
            res_times.append(d["time"])

        avg_finished = np.mean(res_finished)
        total_time = np.sum(res_times)

        print("="*15)
        print(f'Average finished {avg_finished}')
        print(f'Total time spent: {total_time}s')
        print("-"*15)
        print("Finished per test:", res_finished)
        print("Time per test:", res_times)

        return {'finished': avg_finished, 'time': total_time}

    @staticmethod
    def single_test(dispatcher_function,
                    width, height, trains, seed,
                    cities, rails_between_cities, rails_in_city,
                    malfunction_rate, prop_malfunction, min_prop, max_prop,
                    show_map=False):

        if show_map:
            from flatland.utils.rendertools import RenderTool, AgentRenderVariant


        start = time.time()

        speed_ration_map = {1.: 0.25,  # Fast passenger train
                            1. / 2.: 0.25,  # Fast freight train
                            1. / 3.: 0.25,  # Slow commuter train
                            1. / 4.: 0.25}  # Slow freight train

        rail_generator = sparse_rail_generator(max_num_cities=cities,
                                               seed=seed,
                                               grid_mode=False,
                                               max_rails_between_cities=rails_between_cities,
                                               max_rails_in_city=rails_in_city,
                                               )
        schedule_generator = sparse_schedule_generator(speed_ration_map)
        stochastic_data = {'malfunction_rate': malfunction_rate,  # Rate of malfunction occurence of single agent
                           'prop_malfunction': prop_malfunction,
                           'min_duration': min_prop,  # Minimal duration of malfunction
                           'max_duration': max_prop # Max duration of malfunction
                           }
        observation_builder = DummyObservationBuilder()
        env = RailEnv(width=width,
                      height=height,
                      rail_generator=rail_generator,
                      schedule_generator=schedule_generator,
                      number_of_agents=trains,
                      malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                      # Malfunction data generator
                      obs_builder_object=observation_builder,
                      remove_agents_at_target=True
                      # Removes agents at the end of their journey to make space for others
                      )
        env.reset()

        if show_map:
            env_renderer = RenderTool(env, gl="PILSVG",
                                      agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                                      show_debug=False,
                                      screen_height=1920*2,  # Adjust these parameters to fit your resolution
                                      screen_width=1080*2)  # Adjust these parameters to fit your resolution


        dispatcher = dispatcher_function(env)

        max_time_steps = int(4 * 2 * (width + height + 20))

        step = 0
        while True:
            step += 1

            action_dict = dispatcher.step(step)
            next_obs, all_rewards, done, _ = env.step(action_dict)

            if show_map:
                env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

            if done['__all__']:
                break

            if step == max_time_steps:
                break

        if show_map:
            env_renderer.close_window()
            del env_renderer

        finished = np.sum(
            [a.status == RailAgentStatus.DONE or a.status == RailAgentStatus.DONE_REMOVED for a in env.agents])

        finished = finished / len(env.agents)
        elapsed = time.time()-start
        print(f'Trains finished: {finished}. Time spent: {elapsed}s')
        return {'finished': finished, 'time': elapsed}
