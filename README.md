# LocalGenAI
LocalGenAI is a learning exercise that allows me to write code to experiment with text generation LLMs and text to image diffusion models.
As a learning exercise, this is not intended for anything other than prototyping and experimentations.
It provides both a simple .Net / C# API to drive the LLM and diffusion models, and a mini server that accepts OpenAI web APIs.
One of the original reason for developing this, was to allow me to work offline when doing some initial development against OpenAI APIs.
Also, the image generation API for OpenAI is a little more costly and I could save a few dollars by testing / prototyping against a cheaper local model.

From my own experience, a local Stable Diffusion (XL or a 1.5 fine tune like DreamShaper) are quite competitive with DALL.E2 results from OpenAI.
But a local LLM models (anything other than the 70B variants, which my machine could not run) would not hold a candle to GPT3.5 Turbo much less GPT4.
I had to rely on additional few-shot learning prompt engineering to get my local LLM to do something that GPT3.5 Turbo could do in zero shot.

# Dependencies & Setup
1. For LLM, I use llama.cpp as the inference engine. Simply download the binaries for llama.cpp (main.exe and llama.dll) and GGUF models and put them in directories of your choosing and pass in the directories to the API.
2. For diffusion models, you would set up a Python virtual environment, and install all the dependencies (diffusers, transformers, accelerate, torch). The virtual environment folder is passed into the API as a parameter.

# Code Samples
Interface with LLM using the simple C# API. Use llama.cpp as the inference engine.
The chat models generally need correct prompt template to function correctly.
I have examples for a few different prompt styles in the code base.

```C#
// llama-2-13b-chat.Q5_K_M.gguf should be in the model folder
LlamaCpp llamaChat = new Llama2Chat13B(
    @"C:\llm\llama.cpp\bin\cuda12\main.exe",
    @"C:\llm\llama.cpp\model");

llamaChat.StartSession();
string response1 = llamaChat.Chat("tell me about Japan");
llamaChat.EndSession();

// use system message to set up the text generation
llamaChat.StartSession("You are an assistant who speaks like Yoda.");
string response2 = llamaChat.Chat("tell me about Japan");
llamaChat.EndSession();
```
Interface with Stable Diffusion image generation models using simple C# API.
Use Python diffusers library as the inference engine.

```C#
StableDiffusionBase sd = new StableDiffusion15(
    @"C:\python\StableDiffusion");

int seed = Random.Shared.Next();
string filename = Path.Combine(Path.GetTempPath(), $"{seed}.jpg");

sd.TextToImage("a magical computer computer displayed on a stand", filename, seed);

// need initial image and mask as input
sd.Inpaint("a happy Asian blacksmith", "blacksmith.png", "blacksmith_mask.png", filename, seed);

// need initial image as input
sd.ImageToImage("a beutiful female Asian blacksmith", "blacksmith.png", filename, seed, 0.7f);
```
