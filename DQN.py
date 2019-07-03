from agents import agents
import gym
from gym import wrappers
import matplotlib.pyplot as plt
import tensorflow as tf


def make_run_through(env_name,agent,epochs=5):
    env = gym.make(env_name)
    env = wrappers.Monitor(env, './gym-results', force=True)
    agent.set_environment(env)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1),agent=agent.predict)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print(f"restored from {manager.latest_checkpoint}")
    else:
        print("Inititalizing From scratch.")
    for _ in range(epochs):
        observation = agent.reset_environment()
        done = False
        while not done:

            if done:
                env.close()
                break
            else:
                agent.step(observation)
                ckpt.step.assign_add(1)
                if ckpt.step.numpy()%10 ==0:
                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                    print("loss {:1.2f}".format(agent.losses["predict"][-1].numpy()))




if __name__ == "__main__":
    import tensorflow as tf
    # agent = agents.RandomAgent()
    agent = agents.DQN()
    make_run_through("Pong-v0",agent)
    plt.plot(agent.losses["predict"])
    plt.plot(agent.losses["target"])
    plt.show()


