import numpy as np
import gym_sloped_terrain.envs.Laikago_pybullet_env as l
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

tuned_actions_Laikago = np.array([[0.8, 0.8, 0.8, 0.8,
                                   0.428571429, 0.428571429, 0.428571429, 0.428571429,
                                   0.0, 0.0, 0.0, 0.0,
                                   -1.0, -1.0, -1.0, -1.0,
                                   0, 0, 0, 0,
                                   0.0, 0.0, 0.0, 0.0],

                                  [0.8, 0.8, 0.8, 0.8,
                                   0.428571429, 0.428571429, 0.428571429, 0.428571429,
                                   0.0, 0.0, 0.0, 0.0,
                                   -1.0, -1.0, -1.0, -1.0,
                                   0, 0, 0, 0,
                                   0.0, 0.0, 0.0, 0.0],

                                  [0.8, 0.8, 0.8, 0.8,
                                   0.428571429, 0.428571429, 0.428571429, 0.428571429,
                                   0.0, 0.0, 0.0, 0.0,
                                   -1.0, -1.0, -1.0, -1.0,
                                   0, 0, 0, 0,
                                   0.0, 0.0, 0.0, 0.0]
                                  ])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--robotName', help='the robot to be trained for', type=str, default='Laikago')
    parser.add_argument('--policyName', help='file name of the initial policy', type=str, default='IP_')
    args = parser.parse_args()

    if args.policyName == 'IP_':
        args.policyName += args.robotName
    # NUmber of steps per episode
    num_of_steps = 4000

    # list that tracks the states and actions
    states = []
    actions = []
    do_supervised_learning = True

    if args.robotName == 'Laikago':
        # for Laikago
        # idx1 = [3, 0]
        # idx2 = [0, 3]
        idx1 = [0]
        idx2 = [0]
        env = l.LaikagoEnv(render=True, wedge=True, stairs=False, on_rack=False)

        experiment_counter = 0

        for i in idx1:
            for j in idx2:
                if i == 0 and j == 3:
                    break
                t_r = 0
                env.randomize_only_inclines(default=True, idx1=i, idx2=j)

                cstate = env.reset()
                roll = 0
                pitch = 0

                for ii in np.arange(0, num_of_steps):
                    cstate, r, _, info = env.step(tuned_actions_Laikago[experiment_counter])
                    t_r += r
                    states.append(cstate)
                    actions.append(tuned_actions_Laikago[experiment_counter])
                experiment_counter = experiment_counter + 1
                print("Returns of the experiment:", t_r)

    if do_supervised_learning:
        model = LinearRegression(fit_intercept=False)
        states = np.array(states)
        actions = np.array(actions)

        # train
        print("Shape_X_Labels:", states.shape, "Shape_Y_Labels:", actions.shape)
        model.fit(states, actions)
        action_pred = model.predict(states)

        # test
        print('Mean squared error:', mean_squared_error(actions, action_pred))
        res = np.array(model.coef_)
        np.save("./initial_policies/" + args.policyName + ".npy", res)
