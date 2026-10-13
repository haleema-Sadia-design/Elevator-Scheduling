import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Q-Learning settings from the report
FLOORS = 5
ALPHA = 0.1
GAMMA = 0.9
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.05
EPISODES = 500
MAX_STEPS = 100

ACTIONS = ["Move Down", "Wait", "Move Up"]
DELTAS = [-1, 0, 1]


def get_state(elevator, target):
    return (elevator - 1) * FLOORS + (target - 1)


def train_agent(seed=42):
    rng = random.Random(seed)
    q_table = np.zeros((25, 3))
    rewards = []
    epsilon = INITIAL_EPSILON

    for episode in range(EPISODES):
        elevator = rng.randint(1, FLOORS)
        target = rng.randint(1, FLOORS)

        while target == elevator:
            target = rng.randint(1, FLOORS)

        total_reward = 0

        for _ in range(MAX_STEPS):
            state = get_state(elevator, target)

            valid_actions = [
                a for a in range(3)
                if 1 <= elevator + DELTAS[a] <= FLOORS
            ]

            if rng.random() < epsilon:
                action = rng.choice(valid_actions)
            else:
                action = max(
                    valid_actions,
                    key=lambda a: q_table[state, a]
                )

            next_floor = elevator + DELTAS[action]

            reward = -1

            if next_floor == target:
                reward += 10
                next_target = rng.randint(1, FLOORS)

                while next_target == next_floor:
                    next_target = rng.randint(1, FLOORS)
            else:
                next_target = target

            next_state = get_state(next_floor, next_target)

            q_table[state, action] += ALPHA * (
                reward
                + GAMMA * np.max(q_table[next_state])
                - q_table[state, action]
            )

            elevator = next_floor
            target = next_target
            total_reward += reward

        rewards.append(total_reward)
        epsilon = max(MIN_EPSILON, epsilon * 0.99)

    return q_table, rewards


def simulate(q_table, use_q_learning, seed=123, steps=100):
    rng = random.Random(seed)

    elevator = rng.randint(1, FLOORS)
    target = rng.randint(1, FLOORS)

    while target == elevator:
        target = rng.randint(1, FLOORS)

    waiting_times = []
    wait_count = 0
    movement_steps = 0
    requests_served = 0
    requests_started = 1
    total_reward = 0
    history = []

    for step in range(steps):
        state = get_state(elevator, target)

        if use_q_learning:
            valid_actions = [
                a for a in range(3)
                if 1 <= elevator + DELTAS[a] <= FLOORS
            ]
            action = max(
                valid_actions,
                key=lambda a: q_table[state, a]
            )
        else:
            if target > elevator:
                action = 2
            elif target < elevator:
                action = 0
            else:
                action = 1

        next_floor = elevator + DELTAS[action]

        if next_floor != elevator:
            movement_steps += 1

        wait_count += 1
        reward = -1

        if next_floor == target:
            reward += 10
            requests_served += 1
            waiting_times.append(wait_count)
            wait_count = 0

            target = rng.randint(1, FLOORS)
            while target == next_floor:
                target = rng.randint(1, FLOORS)

            requests_started += 1

        elevator = next_floor
        total_reward += reward

        history.append({
            "Step": step + 1,
            "Elevator Floor": elevator,
            "Requested Floor": target,
            "Action": ACTIONS[action],
            "Reward": reward
        })

    average_wait = (
        np.mean(waiting_times) if waiting_times else 0
    )

    return {
        "Controller": "Q-Learning" if use_q_learning
        else "Rule-Based",
        "Average Waiting Time": average_wait,
        "Movement Steps": movement_steps,
        "Cumulative Reward": total_reward,
        "Requests Served": requests_served,
        "Requests Started": requests_started,
        "Service Rate (%)": (
            requests_served / requests_started * 100
        )
    }, pd.DataFrame(history)


st.set_page_config(
    page_title="Smart Elevator Scheduling",
    page_icon="🛗",
    layout="wide"
)

st.title("🛗 Smart Elevator Scheduling Using Q-Learning")
st.write(
    "Train an elevator agent for a five-floor building and "
    "compare it with a rule-based controller."
)

st.sidebar.header("Simulation Settings")

episodes = st.sidebar.selectbox(
    "Training Episodes",
    [500, 1000, 1500, 2000],
    index=0
)

evaluation_steps = st.sidebar.number_input(
    "Evaluation Steps",
    min_value=20,
    max_value=500,
    value=100
)

seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    value=42
)

st.subheader("Q-Learning Parameters")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Learning Rate α", ALPHA)
c2.metric("Discount Factor γ", GAMMA)
c3.metric("Initial ε", INITIAL_EPSILON)
c4.metric("Minimum ε", MIN_EPSILON)
c5.metric("Floors", FLOORS)

if st.button("Train Agent and Generate Graphs", type="primary"):

    with st.spinner("Training and running simulations..."):

        original_episodes = EPISODES
        EPISODES = int(episodes)

        q_table, training_rewards = train_agent(int(seed))

        EPISODES = original_episodes

        q_result, q_history = simulate(
            q_table, True, int(seed) + 100,
            int(evaluation_steps)
        )

        baseline_result, baseline_history = simulate(
            q_table, False, int(seed) + 100,
            int(evaluation_steps)
        )

        results = pd.DataFrame([q_result, baseline_result])

        st.session_state["q_table"] = q_table
        st.session_state["rewards"] = training_rewards
        st.session_state["results"] = results
        st.session_state["history"] = pd.concat(
            [
                q_history.assign(Controller="Q-Learning"),
                baseline_history.assign(Controller="Rule-Based")
            ],
            ignore_index=True
        )

if "results" in st.session_state:

    results = st.session_state["results"]
    rewards = st.session_state["rewards"]
    q_table = st.session_state["q_table"]
    history = st.session_state["history"]

    st.success("Training and simulation completed.")

    st.subheader("Performance Comparison")
    st.dataframe(
        results.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Graph 1: Training Rewards")

    fig, ax = plt.subplots()
    ax.plot(rewards, alpha=0.4, label="Episode Reward")
    ax.plot(
        pd.Series(rewards).rolling(
            20, min_periods=1
        ).mean(),
        label="20-Episode Average"
    )
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Q-Learning Training Progress")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Graph 2: Average Waiting Time")

    fig, ax = plt.subplots()
    ax.bar(
        results["Controller"],
        results["Average Waiting Time"]
    )
    ax.set_ylabel("Waiting Time (Steps)")
    ax.set_title("Average Passenger Waiting Time")
    ax.grid(axis="y")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Graph 3: Elevator Movement")

    fig, ax = plt.subplots()
    ax.bar(
        results["Controller"],
        results["Movement Steps"]
    )
    ax.set_ylabel("Movement Steps")
    ax.set_title("Total Elevator Movement")
    ax.grid(axis="y")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Graph 4: Successful Service Rate")

    fig, ax = plt.subplots()
    ax.bar(
        results["Controller"],
        results["Service Rate (%)"]
    )
    ax.set_ylabel("Service Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Passenger Request Service Rate")
    ax.grid(axis="y")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Learned Q-Table")

    q_rows = []

    for floor in range(1, FLOORS + 1):
        for target in range(1, FLOORS + 1):
            state = get_state(floor, target)

            q_rows.append({
                "Elevator Floor": floor,
                "Requested Floor": target,
                "Move Down": q_table[state, 0],
                "Wait": q_table[state, 1],
                "Move Up": q_table[state, 2],
                "Best Action": ACTIONS[
                    np.argmax(q_table[state])
                ]
            })

    q_df = pd.DataFrame(q_rows)
    st.dataframe(q_df.round(2), use_container_width=True)

    st.subheader("Simulation History")
    st.dataframe(history, use_container_width=True)

    st.download_button(
        "Download Results CSV",
        results.to_csv(index=False),
        file_name="elevator_results.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download Q-Table CSV",
        q_df.to_csv(index=False),
        file_name="elevator_q_table.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download Simulation History CSV",
        history.to_csv(index=False),
        file_name="elevator_history.csv",
        mime="text/csv"
    )

else:
    st.info(
        "Click the training button to generate graphs and results."
    )

st.divider()
st.caption(
    "Educational simulation only. This program does not control "
    "a real elevator. Results depend on the simulation and reward design."
)