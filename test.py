import birdseyelib as bird
import time
HOST = "127.0.0.1"
PORT = 8080
if __name__ == "__main__":
    
    
    client = bird.Client(HOST, PORT)
    memory = bird.Memory()
    controller_input = bird.ControllerInput()
    emulation = bird.Emulation()
    external_tool = bird.ExternalTool()


    client.connect()
    print("Connecting to server at {} on port {}.".format(HOST, PORT))
   
    
    while client.is_connected():
        controller_input.set_controller_input(right=True)
        client.send_requests([memory, controller_input, emulation])
        print(client.get_responses())
    print("Could not connect to external tool :[")
