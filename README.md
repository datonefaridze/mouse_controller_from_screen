<h1># Mouse Controller:</h1>
<h3>Aim of this repository is to create software that enables humans to interact with computer without mouse.</h3>

<div>
	<p>
		Repo contains script to run training and script which is responsible for user to interact with computer without mouse
	</p>
	<p>model was trained on 60 epochs and It showed very high accuracy so I decided to not continue training after 60 epochs</p>
</div>
<div>
	<h1>
		How to interact with computer:
	</h1>
	<p>once you run the mosue_from_screen.py script you have to select finger which you want to track, after that you can take your finger and slide it across the air, at that time mouse will follow your finger directions. when you want to click on something you just have to show 5 fingers to the camera, if you show nothing or your fist it doesn't generate click, if you show 5 fingers it will generate click.</p>
	
</div>

<div>
	<h1>Future Plans:</h1>
	<p>Add action recognition to the code, currently it clicks if you show your five fingers, it would be more convinient if it would generate click on action like you do on smartphone screen. also this repo is tested on Windows, so in future i want to use Docker to make it accessible to all users</p>

</div>
<div>
	<h1>Here is demo how it performs:</h1>
</div>


![grab-landing-page](https://github.com/datonefaridze/mouse_controller_from_screen/blob/main/demo_giff.gif)
