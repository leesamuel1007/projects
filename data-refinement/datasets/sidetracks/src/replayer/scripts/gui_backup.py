import tkinter as tk
import os
from PIL import Image, ImageTk
import rospy
import json
import os

class GUIInterface:
    def __init__(self):
        rospy.init_node("replayer_gui")
        self.root = tk.Tk()
        self.root.title("Replayer")
        # self.root.geometry("800x600")
        # self.root.resizable(False, False)
        self.frame = tk.Frame(self.root, padx=20, pady=20)
        self.frame.grid(row=0, column=0)

        # flags to control the annotation
        self.test_controller_start_flag = False
        self.pause_start_flag = False
        self.alignment_start_flag = False

    def start(self):
        this_file_path = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(this_file_path, "images")
        image_paths = ["pause.png", "resume.png", "backward.png", "test_controller.png", "wait.png", "alignment.png","phase_change.png", "false_control.png"]
        button_labels = ["pause","play forward","play backward","test controller","pausing"]
        self.buttons = []
        maxsize = (100, 100)
        input_font = ('helvetica', 10)
        # text boxes for false control annotation
        self.false_action_input = tk.Text(self.frame, width=50, height=5) 
        
        self.false_action_input.grid(row=0, column=0, columnspan=2)
        
        # false_action_input.pack() 
        self.correct_action_input = tk.Text(self.frame, width=50,height=5) 
        self.correct_action_input.grid(row=0, column=2, columnspan=2)
        
        # correct_action_input.pack() 
        # label_text_1 = tk.StringVar()
        # label_text_1.set("What is the correct action?")
        label_font = ('helvetica', 12, 'bold')
        lbl = tk.Label(self.frame, text = "What is the false action?", font=label_font)
        lbl.grid(row=1, column=0, columnspan=2, padx=5, pady=10)

        lbl_2 = tk.Label(self.frame, text = "What is the correct action to be?", font=label_font)
        lbl_2.grid(row=1, column=2, columnspan=2, padx=5, pady=10)


        self.before_action_input = tk.Text(self.frame, width=50,height=5) 
        self.before_action_input.grid(row=2, column=0, columnspan=2)

        self.after_action_input = tk.Text(self.frame, width=50,height=5)
        self.after_action_input.grid(row=2, column=2, columnspan=2) 

        lbl_3 = tk.Label(self.frame,text="before this behavior (error), what is the participant doing?",font=label_font)
        lbl_3.grid(row=3, column=0, columnspan=2, padx=5, pady=10)
        lbl_4 = tk.Label(self.frame,text="after this behavior (error), what is the participant doing?",font=label_font)
        lbl_4.grid(row=3, column=2, columnspan=2, padx=5, pady=10)

        self.description_input = tk.Text(self.frame, width=50,height=5)
        self.description_input.grid(row=2, column=4, columnspan=2) 
        lbl_5 = tk.Label(self.frame,text="description of the behavior",font=label_font)
        lbl_5.grid(row=3, column=4, columnspan=2, padx=5, pady=10)
        # add the buttons
        for i, image_path in enumerate(image_paths, start=1):
            image_path = os.path.join(image_dir, image_path)
            image = Image.open(image_path)
            r1 = image.size[0]/maxsize[0] # width ratio
            r2 = image.size[1]/maxsize[1] # height ratio
            ratio = max(r1, r2)
            newsize = (int(image.size[0]/ratio), int(image.size[1]/ratio))

            image = image.resize(newsize, Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)

            button = tk.Button(self.frame, image=photo, width=100, height=100, command=lambda i=i: self.on_click(i))
            button.image = photo
            self.buttons.append(button)

        # add the buttons to the frame except the false_control button
        for i, button in enumerate(self.buttons[:-1]):
            # pass
            # button.pack(side=tk.LEFT, padx=10, pady=20)
            button.grid(row=4, column=i, padx=10, pady=20)

        # place the false_control button at the right of text boxes
        self.buttons[-1].grid(row=0, column=4, padx=10, pady=20)

        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)
        self.root.mainloop()


    def on_click(self, i):
        if i == 1:
            rospy.loginfo("pause replay!")
            rospy.set_param("/replay/playflag", 0)
        if i == 2:
            rospy.loginfo("start replay forward!")
            rospy.set_param("/replay/playflag", 1)
        if i == 3:
            rospy.loginfo("start replay backward!")
            rospy.set_param("/replay/playflag", -1)
        
        # annotation: test controller
        if i == 4:
            if not self.test_controller_start_flag:
                rospy.loginfo("annotate info: test controller")
                # need to decrement the current action index by 1 because the current_action_index actually next action index
                self.current_action_step_start = rospy.get_param("/replay/current_action_index", 0) 
                # change the button color to green
                self.buttons[3].config(bg="green")
                self.test_controller_start_flag = True
            else:
                rospy.loginfo("annotate info: test controller end")
                self.current_action_step_end = rospy.get_param("/replay/current_action_index", 0) 
                current_user_id = rospy.get_param("/replay/bag_number", 0)
                before_action = self.before_action_input.get("1.0", "end-1c")
                after_action = self.after_action_input.get("1.0", "end-1c")
                description = self.description_input.get("1.0", "end-1c")
                info = dict(
                    user_id=current_user_id,
                    action_step_start=self.current_action_step_start,
                    action_step_end=self.current_action_step_end,
                    actions_before_this_behavior = before_action,
                    actions_after_this_behavior = after_action,
                    behavior_description = description,
                    annotation="test controller"
                )
                self.before_action_input.delete("1.0", "end")
                self.after_action_input.delete("1.0", "end")
                self.description_input.delete("1.0", "end")
                # change the button color back to grey
                self.buttons[3].config(bg="grey")
                self.test_controller_start_flag = False
                json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),f"annotations/user_{current_user_id}.json")
                with open(json_path, "a+") as f:
                    json.dump(info, f, indent=4)
                    f.write("\n")
        
        if i == 5:
            if not self.pause_start_flag:
                rospy.loginfo("annotate info: pause")
                # need to decrement the current action index by 1 because the current_action_index actually next action index
                self.current_action_step_start_pause = rospy.get_param("/replay/current_action_index", 0)
                # change the button color to green
                self.buttons[4].config(bg="green")
                self.pause_start_flag = True
            else:
                rospy.loginfo("annotate info: pause end")
                self.current_action_step_end_pause = rospy.get_param("/replay/current_action_index", 0)
                current_user_id = rospy.get_param("/replay/bag_number", 0)
                before_action = self.before_action_input.get("1.0", "end-1c")
                after_action = self.after_action_input.get("1.0", "end-1c")
                description = self.description_input.get("1.0", "end-1c")
                info = dict(
                    user_id=current_user_id,
                    action_step_start=self.current_action_step_start_pause,
                    action_step_end=self.current_action_step_end_pause,
                    actions_before_this_behavior = before_action,
                    actions_after_this_behavior = after_action,
                    behavior_description = description,
                    annotation="pause"
                )
                self.before_action_input.delete("1.0", "end")
                self.after_action_input.delete("1.0", "end")
                self.description_input.delete("1.0", "end")
                # change the button color back to grey
                self.buttons[4].config(bg="grey")
                self.pause_start_flag = False
                json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),f"annotations/user_{current_user_id}.json")
                with open(json_path, "a+") as f:
                    json.dump(info, f, indent=4)
                    f.write("\n")
        
        if i == 6:
            # alignment annotation
            if not self.alignment_start_flag:
                rospy.loginfo("annotate info: alignment")
                # need to decrement the current action index by 1 because the current_action_index actually next action index
                self.current_action_step_start_alignment = rospy.get_param("/replay/current_action_index", 0)
                # change the button color to green
                self.buttons[5].config(bg="green")
                self.alignment_start_flag = True
            else:
                rospy.loginfo("annotate info: alignment end")
                self.current_action_step_end_alignment = rospy.get_param("/replay/current_action_index", 0)
                current_user_id = rospy.get_param("/replay/bag_number", 0)
                before_action = self.before_action_input.get("1.0", "end-1c")
                after_action = self.after_action_input.get("1.0", "end-1c")
                description = self.description_input.get("1.0", "end-1c")
                info = dict(
                    user_id=current_user_id,
                    action_step_start=self.current_action_step_start_alignment,
                    action_step_end=self.current_action_step_end_alignment,
                    actions_before_this_behavior = before_action,
                    actions_after_this_behavior = after_action,
                    behavior_description = description,
                    annotation="alignment"
                )
                self.before_action_input.delete("1.0", "end")
                self.after_action_input.delete("1.0", "end")
                self.description_input.delete("1.0", "end")

                # change the button color back to grey
                self.buttons[5].config(bg="grey")
                self.alignment_start_flag = False
                json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),f"annotations/user_{current_user_id}.json")
                with open(json_path, "a+") as f:
                    json.dump(info, f, indent=4)
                    f.write("\n")


        if i == 7:
            # phase change annotation
            before_action = self.before_action_input.get("1.0", "end-1c")
            after_action = self.after_action_input.get("1.0", "end-1c")
            current_user_id = rospy.get_param("/replay/bag_number", 0)
            action_step_change = rospy.get_param("/replay/current_action_index", 0)
            info = dict(
                user_id = current_user_id,
                step = action_step_change,
                actions_before_this_behavior = before_action,
                actions_after_this_behavior = after_action,
                annotation = "task phase change"
            )
            json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),f"annotations/user_{current_user_id}.json")
            with open(json_path, "a+") as f:
                json.dump(info, f, indent=4)
                f.write("\n")
            self.before_action_input.delete("1.0", "end")
            self.after_action_input.delete("1.0", "end")

        if i == 8:
            # false control annotation
            false_action = self.false_action_input.get("1.0", "end-1c")
            correct_action = self.correct_action_input.get("1.0", "end-1c")
            current_user_id = rospy.get_param("/replay/bag_number", 0)
            action_step_false = rospy.get_param("/replay/current_action_index", 0)
            before_action = self.before_action_input.get("1.0", "end-1c")
            after_action = self.after_action_input.get("1.0", "end-1c")
            description = self.description_input.get("1.0", "end-1c")
            info = dict(
                user_id=current_user_id,
                false_action=false_action,
                correct_action=correct_action,
                actions_before_this_behavior = before_action,
                actions_after_this_behavior = after_action,
                step=action_step_false,
                behavior_description = description,
                annotation="false control"
            )
            json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),f"annotations/user_{current_user_id}.json")
            with open(json_path, "a+") as f:
                json.dump(info, f, indent=4)
                f.write("\n")
            rospy.loginfo("annotate info: false control")
            self.false_action_input.delete("1.0", "end")
            self.correct_action_input.delete("1.0", "end")
            self.before_action_input.delete("1.0", "end")
            self.after_action_input.delete("1.0", "end")
            self.description_input.delete("1.0", "end")

if __name__ == "__main__":
    gui = GUIInterface()
    gui.start()