from pynput.keyboard import Key, Listener, KeyCode

class KeyListener:

    key_mapping = {KeyCode.from_char('a'): "LEFT", KeyCode.from_char('d'): "RIGHT", KeyCode.from_char('w'): "UP", KeyCode.from_char("s"): "DOWN", Key.space: "FIRE"}

    def __init__(self, env):
        # meaning = env.unwrapped.get_action_meanings()
        # self.meaning_to_action = dict(zip(meaning, list(range(len(meaning)))))
        # self.time_between_frames = 0.1
        # self.current_act = self.meaning_to_action["NOOP"]
        pass

    def on_press(self, key):
        print(key)
        print(key == KeyCode.from_char("a"))
        print(key == Key.space)
        # if key in self.key_mapping:
        #     self.current_act = self.meaning_to_action[self.key_mapping[key]]
        # elif key == "+":
        #     self.time_between_frames /= 2
        # elif key == "-":
        #     self.time_between_frames *= 2
    def on_release(self, key):
        # if key in self.key_mapping:
        #     self.current_act = self.meaning_to_action["NOOP"]
        if key == Key.esc:
            return False

key_listener = KeyListener(None)
Listener(on_press=key_listener.on_press, on_release=key_listener.on_release).start()

while True:
    pass
