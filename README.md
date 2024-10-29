# checkbin-virtual-staging-demo
Demo of Checkbin at work in a virtual staging AI app!

Virtual staging apps aim to take an input image of a home, and aim to redesign it in a new style. Our example pipeline has three parts: 
Part 1: Segment the image
Part 2: Remove the furniture
Part 3: Generate an image in the new style! 


### Step 1 - Tools 🛠️
To run this demo code, you'll need auth tokens (and accounts) from the following services:
- **[Modal](www.modal.com)** - We use Modal to run the training and inference script on cloud GPUs. You can get a Modal token by signing up [here](https://modal.com/signup).
- **[Checkbin](www.checkbin.dev)** - We use Checkbin ✅🗑️ to compare the results of different models. You can get a Checkbin token by signing up [here](www.checkbin.dev/signup).

### Step 2 - Create your [Checkbin app](https://app.checkbin.dev/dashboard/apps)

Select the 'New App' button on the [Checkbin app page](https://app.checkbin.dev/dashboard/apps). Name it something descriptive! I went for 'virtual_staging_demo'. 

![create-app](https://syntheticco.blob.core.windows.net/virtual-staging-demo/virtual_staging_demo_create_app.gif)

### Step 3 - Create your Input set!

Before you run anything, you'll need a some Input examples to run through the virtual staging pipeline! For virtual staging the inputs have two pieces:

- **image_url** (ex. ) - this is a URL to the image that you will be redesigning!
- **prompt** (ex. ") - this is the prompt that the AI will use to redesign your room!

For my demo, I downloaded 20 random images off of AirBnB listings and generated prompts with ChatGPT. These examples are available in this `test_inputs.json` file. You can use this or replace it with your own!

You can use your JSON file to create a Checkbin InputSet, either in code or in the dashboard.

To create your InputSet in code, uncomment the commented out lines in `run_test.py`:

```
# TODO: uncomment this to create a new input set from code!
# checkbin_input_set = checkbin_app.create_input_set("20x Test Inputs")
# input_set_id = checkbin_input_set.create_from_json(json_file='./test_inputs.json')
# print(input_set_id) # Store the input_set_id to use in the 
```

To create your InputSet from the dashboard, navigate to the [Checkbin Input Page](https://app.checkbin.dev/dashboard/input-sets) and upload your JSON file! Make sure you've [selected the right app]((https://app.checkbin.dev/dashboard/apps))first! 

After creating your InputSet, copy the set_id! 

![create-input-set](https://syntheticco.blob.core.windows.net/virtual-staging-demo/virtual_staging_demo_input_set_creation.gif)


### Step 4 - Run the demo!

After creating your InputSet, you're ready to run the pipeline! Replace the following lines of code in `run_test.py` with your input_set_id:

```
# TODO: replace this with the input set ID you create!
input_set_id = "bcbfb43d-c194-435d-8dba-b21b7705149e" 
```

Then you can run the pipeline on Modal: 

```
modal run run_test.py
```

Once the pipeline is started, you should see it running on the [Checkbin Runs Page](https://app.checkbin.dev/dashboard/runs)! Hover over the row and click "view in grid" to view your visualization!

![view-run](https://syntheticco.blob.core.windows.net/virtual-staging-demo/virtual_staging_demo_view_run_trimmed.gif)

That's it! You've successfully run your test inputs through the pipeline. In the Checkbin grid, you'll be able to see the intermediate images as your inputs run through the steps of our virtual staging app!

### Acknowledgments

This demo uses an adapted version of Lavreniuk's [Generative Interior Design](https://github.com/Lavreniuk/generative-interior-design) code. Thanks to him for his contributions!