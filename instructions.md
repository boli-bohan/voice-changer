# Voice Changer Project

### Lets try building a real-time voice changing app!

This is a fun project designed to build out the infrastructure for a hypothetical voice-changing app. It’s not a real workload, but the short timeframe + full stack nature of the project should hopefully be entertaining and give you a feel for both the pace and nature of work we have here at Spellbrush. Let’s get to it!

# Background

Lets pretend that our research team has delivered a streaming audio model that can convert input audio to output audio in a real-time fashion. In our case, maybe it takes in streaming microphone input and converts it into a vocaloid anime voice.

---

![image.png](attachment:2e9dfc2a-426d-479f-b7db-fa4333408f5a:image.png)

Users love the model in testing, so now it’s time to build out the infrastructure!

# Building the Local Version

We’ll need a few mock stub applications to fill in. Potentially the ML team can have something ready later, but to make life easier let’s control the full stack.

1. We’ll need a python application that accepts streaming audio, and transforms it somehow. For now, lets try a simple pitch-shift operation.
2. We’ll need a user interface! Let’s build a super simple one in react or even pure javascript. We really only need a start/stop button. And maybe a way to visualize the waveform so we can see something happening.

    ![image.png](attachment:30c4f807-ead3-40e3-9196-e09f8f774ddc:image.png)

3. Now we need the frontend to stream the input audio to the python backend, and also receive back the transformed audio. There are a few ways of doing this, WebRTC, Websockets, etc. Pick whatever works and is easy.

Once we have all these in place, we should be able to hit start on our little frontend app, connect to [`localhost`](http://localhost) and then hear back the resulting pitch-shifted audio!

# Scaling it Up

Now it’s time to scale up the system so we can serve it at scale to the real world. Let’s build a sticky autoscaling solution that is able to handle the load while maintaining the requirements on our model.

- This model is heavy, so we want to try running it on the cloud (e.g. maybe our users are on mobile and the computation is complex.
- Because the model is streaming, sessions are inherently stateful — the past few tokens of audio + the local KV-cache and audio state are loaded into each GPU, so we cannot arbitrarily rebalance users between machines without them experiencing a drop in connection.
- We’ve (hypothetically) benchmarked our usage case, and each GPU can service exactly 4 users at a time. More than that will cause the GPU to run out of VRAM.
- We can only have a ***single type*** of model loaded per GPU. So if we want a Teto model, it must be on a different GPU than the Miku model.

To simulate the above, modify the local python application such that it takes 10 seconds before it’s ready to start streaming audio — this will simulate the delay required to load the model into VRAM.

So we want a backend that is able to intelligently:

- Connect a user to an available GPU server serving the correct model.
- Spin up a new GPU server if there are more users than there are available model serving instances.
- Minimize the total latency for connecting to the system across all users.
- Minimize our total GPU bill (these things are super expensive!)

![image.png](attachment:f3d69370-3569-41f2-ad40-41a98ad2efbb:image.png)

<aside>
💡

There a decision here to use either a central orchestrator which would allow auto-scaling and more efficient bin-packing of jobs, or a more distributed intelligence, e.g. choreography pattern. Both have their pros & cons!

</aside>

There are no specific requirements on the stack or implementation, but I do have several nice-to-have features:

- It should be easy to deploy the infrastructure.
- The system should feel responsive even when under load.

    <aside>
    💡

    How should we communicate to users that their model is booting up?

    </aside>

- In general, excess complexity should be avoided. It should be easy for any new engineer to fully internalize the workings of the system, and make changes without needing to understand massive stateful k8s systems.
- It should be also be easy to add new models, or push new updates to models in production. Maybe we have `miku` and `miku-dev`? The easier it is for the research team to deploy, the better the product is as we can iterate quickly.

    <aside>
    💡

    What is our draining behavior when we push an update to a model?

    </aside>


Once we have this in place, we’re ready to move on to dealing with the last bit of GPU complexity — machine crashes!

# A Simple Accounting System

We can’t let just anyone connect to the system, we want to ensure a few things:

1. That an user cannot have multiple streams open, thus wasting our resources.
2. That we properly tally up how many minutes a user has spent using our super cool voice conversion software — accounting wants to knows

We don’t need to build a full login / logout system, that’s outside the scope of this project. But we can just have a user id somewhere: maybe in the connection parameter, maybe as a cookie, and then use that as the ground truth for the accounting system.

# Making it Fault Tolerant

GPU systems are notoriously flakey, and we have a very stateful system here, so we should now try to make sure our solution handles errors.

Add in a probabilistic  crash to the python application that errors out in a few seconds, and see if we’re able to correctly rebalance.

Now see if our system can correctly handle this case:

- Are we able to rebalance users to other servers if the model crashes?
- Can we continue to keep our connection latency low?
- Do we correctly account for how much time a user has used our service?

Try crashing an entire region and seeing whether our characteristics hold.

# Making it Global

We want the lowest latency possible, so ideally we are able to boot up machines *close* to where a user is. RTT between San Francisco and Tokyo can be as as high as 150ms, which introduces a very audible delay.

We should modify our orchestration system such that we can boot up sticky machines in the region *closest* to the user so that total round-trip latency is minimized.

Modify the infrastructure so that we can bring up GPU machines in random regions and have them correctly serve clients.

<aside>
💡

Can we go multi-region without significantly complicating our deployment?

What about introducing GPUs from different service providers? Often we’ll have spare capacity in random datacenters which we can buy for cheap, and being able to throw those into the mix is very cost effective, but those can’t be managed via AKS or GKE.

</aside>

# Final Notes

Many of the above points are underspecced intentionally to give you the freedom to design and implement whatever you want! If you have any questions, or want to bounce around a few candidate designs, feel free to ask the team!

Remember, as a small team and startup, our #1 advantage is velocity. So ideally any system you design should be focused on creating infrastructure that enables high shipping velocity.

- Extremely easy developer experience. The system must be easy for all three independent teams to iterate and improve on (incl. infrastructure, research, and app development)
- Low total complexity: the system must be maintainable and understandable a single individual.
- We are willing to make tradeoffs for devx over full system correctness. e.g. it is OK if we miss accounting 30minutes of product use in exchange for the system to be deployable and maintainable by a single person.
- We are currently attempting to consolidate language use within the company to python, typescript, and c#. It is not necessary to use these, but does make life a lot easier for others to understand the codebase.
