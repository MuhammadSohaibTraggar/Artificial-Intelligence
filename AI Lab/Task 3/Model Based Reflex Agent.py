class ModelBasedReflexAgent:
    def __init__(self, desired_temperature):
        self.desired_temperature = desired_temperature
        self.previous_states = {}

    def perceive(self, room, current_temperature):
        return current_temperature

    def decide(self, room, current_temperature):
        previous_action = self.previous_states.get(room, None)
        
        if current_temperature < self.desired_temperature:
            action = "Turn on heater"
        else:
            action = "Turn off heater"
      
        if action == previous_action:
            return f"No action needed (heater already {action.split()[-1]})."
        else:
            return action

    def updateModel(self, room, action):
        self.previous_states[room] = action

    def act(self, room, current_temperature):
        current_temperature = self.perceive(room, current_temperature)
        action = self.decide(room, current_temperature)
        self.updateModel(room, action)
        return action
def main():
    rooms = {
        "Living Room": 18,
        "Bedroom": 22,
        "Kitchen": 20,
        "Bathroom": 24
    }

    desired_temp = 22
    agent = ModelBasedReflexAgent(desired_temp)

    print("First cycle:")
    for room, temp in rooms.items():
        action = agent.act(room, temp)
        print(f"{room}: Current temp: = <---- {temp}°C ----> {action}.")

if __name__ == "__main__":
    main()