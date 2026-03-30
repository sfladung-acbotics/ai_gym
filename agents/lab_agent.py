"""This is an llm based agent for controlling lab equipment."""

import ollama
from faster_whisper import WhisperModel
import queue
import threading


class Lab_State:
    def __init__(self):
        self.tool_map = {
            "set_power_supply_voltage": self.set_power_supply_voltage,
            "set_power_supply_current": self.set_power_supply_current,
            "read_power_supply_current_measurement": self.read_power_supply_current_measurement,
            "read_power_supply_current_limit": self.read_power_supply_current_limit,
        }
        self.tools = list(self.tool_map.values())
        self.current_limit = 3

    def set_power_supply_voltage(self, voltage: str):
        """
        Set the voltage of the power supply.
        Args:
            voltage: Target voltage to set.
        """
        v = float(voltage.strip("V"))
        print("Would set power supply to: %f" % (v,))
        return "Success: Voltage set to %f" % (v,)

    def set_power_supply_current(self, current: str):
        """
        Set the current of the power supply. This is an absolute, not relative value.
        Args:
            current: Target current to set.
        """
        cur = float(current.strip("A"))
        self.current_limit = cur
        print("Would set power supply to: %f" % (cur,))
        return "Success: Current limit set to %f" % (cur,)

    def read_power_supply_current_measurement(self):
        """
        Read the current measurement from the power supply.
        """
        print("Attempting to read current")
        return "2.1A"

    def read_power_supply_current_limit(self):
        """
        Read the current limit set from the power supply. This is also the current setting.
        """
        print("Attempting to read current limit")
        return str(self.current_limit) + "A"


class Lab_Agent:
    def __init__(self, speech_agent=None):
        self.lab = Lab_State()
        self.commands = queue.Queue()
        self.speech_agent = speech_agent

    def add_command(self, cmd):
        self.commands.put(cmd)

    def run_once(self):
        try:
            command = self.commands.get(timeout=1)
        except queue.Empty:
            return
        final_response = ""
        # 2. Start the chat with the tool enabled
        messages = [
            {
                "role": "system",
                "content": "You are a terse assistant with access to tools. If a tool is relevant, you MUST provide a tool call in the correct XML or JSON format. Do not just think about it. If more than 1 tool call would be required, output the first and terminate. You will be rerun with it's result. Do not guess. Use tools to get needed information. Be terse, DO NOT prompt the user if they want additional adjustments or verifications",
            },
            {"role": "user", "content": command},
        ]
        for _ in range(5):
            # print(messages)
            response = ollama.chat(
                model="qwen3:1.7b",
                messages=messages,
                options={
                    "temperature": 0,  # 0 is the most deterministic
                    "seed": 42,  # Optional: lock the random seed for even more consistency
                },
                tools=self.lab.tools,  # Pass the function here
            )
            final_response = response
            messages.append(response.message)
            # print(response)
            if response.message.tool_calls is None:
                break
            for call in response.message.tool_calls:
                function_to_call = self.lab.tool_map[call.function.name]
                result = function_to_call(**call.function.arguments)

                # CRITICAL: Feed the result back to the model as a 'tool' role
                messages.append(
                    {
                        "role": "tool",
                        "content": str(result),
                        "name": call.function.name,
                    }
                )
                print(f"Executed {call.function.name}: {result}")
        print(final_response.message.content)
        if self.speech_agent:
            self.speech_agent.add_speech(
                final_response.message.content,
                "The message refers to electronics. Unit abbreviations are likely A for Amps, V for Volts etc",
            )

    def run(self):
        while True:
            self.run_once()

    def run_as_thread(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()


if __name__ == "__main__":
    import voice_agent
    import os

    VOICE_MODEL = "en_US-lessac-medium.onnx"
    VOICE_CONFIG = "en_US-lessac-medium.onnx.json"

    if os.path.exists(VOICE_MODEL):
        tts = voice_agent.TTS_Engine(model_path=VOICE_MODEL, config_path=VOICE_CONFIG)
    else:
        print(f"Warning: Voice model not found at {VOICE_MODEL}. TTS disabled.")
        tts = None
    voice_output_parser = voice_agent.Voice_Parse_Agent(tts=tts)
    agent = Lab_Agent(speech_agent=voice_output_parser)
    voice_agent = voice_agent.Voice_Agent(output_callbacks=(agent.add_command,))
    voice_agent.run()
    agent.run_as_thread()
    voice_output_parser.run_as_thread()
    while True:
        cmd = input("Next Command: ")
        agent.add_command(cmd)
