import room

room_size = 5
num_tasks = 2
work_per_task = 3

env = room.Room(room_size, num_tasks, work_per_task)
env.render()
while True:
    action = input("pick action: ")
    action = int(action)
    obs, reward, done, info = env.step(action)
    env.render()
    print(reward)
