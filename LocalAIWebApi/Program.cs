using System.Text.Json;
using System.Text.Json.Serialization;

using OpenAiWebApi;
using StableDiffusionAdapter;
using Llama2Adapter;

internal class Program
{
    // this config allows to reuse same pipeline for multiple
    // requests of the same kind (t2i, i2i, inpaint)
    private const bool KeepStableDiffusionRunning = true;
    private const bool KeepLlamaRunning = true;
    private static Type LlmModel = typeof(Llama2Chat13B);
    private static Type SdModel = typeof(StableDiffusion15);
    private const float Temperature = 0.7f;

    private static StableDiffusionBase? _stableDiffusion = null;
    private static LlamaCpp? _llama = null;

    private static void Main(string[] args)
    {
        WebApplicationBuilder builder = WebApplication.CreateBuilder(args);

        builder.Services.ConfigureHttpJsonOptions(options =>
        {
            options.SerializerOptions.PropertyNamingPolicy = 
                JsonNamingPolicy.CamelCase;
            options.SerializerOptions.Converters.Add(
                new JsonStringEnumMemberConverter(JsonNamingPolicy.CamelCase));
            options.SerializerOptions.DefaultIgnoreCondition = 
                JsonIgnoreCondition.WhenWritingNull;
        });

        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddSwaggerGen();

        WebApplication app = builder.Build();

        // Configure the HTTP request pipeline.
        if (app.Environment.IsDevelopment())
        {
            app.UseSwagger();
            app.UseSwaggerUI();
        }

        /* OpenAI chat completion API is stateless, but for llama2 
         * inference, we need to consider making the code stateful to 
         * avoid having to reload model for every additional prompt
         */
        app.MapPost("/v1/chat/completions", (ChatCreateRequest request) =>
        {
            int seed = Random.Shared.Next();

            // valid incoming Messages
            // 1. user
            // 2. system + user
            // 3. system + (user + assistant) x 1 or more + user

            string newMessage = request.Messages
                .Last(message => message.Role == ChatRole.User).Content;
            List<ChatMessage>? contextMessages = null;
            if (request.Messages.Length > 1)
            {
                contextMessages = request.Messages.SkipLast(1).ToList(); 
            }

            LlamaCpp llamaChat = EnsureLlamaInstance(contextMessages);

            string aiResponse = llamaChat.Chat(newMessage);
            if (!KeepLlamaRunning)
            {
                llamaChat.EndSession();
            }

            ChatCompletionResponse response = new()
            {
                Id = "cmpl-" + Random.Shared.Next().ToString(),
                Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                Object = "text_completion",
                Model = request.Model,
                Choices = new ChatCompletionChoice[]
                {
                    new ChatCompletionChoice()
                    {
                        Index = 0,
                        Message = new ChatMessage()
                        {
                            Role = ChatRole.Assistant,
                            Content = aiResponse
                        },
                        FinishReason = FinishReason.Stop
                    }
                }
            };

            return response;
        }).Accepts<ChatCreateRequest>("application/json")
            .Produces<ChatCompletionResponse>(200);

        app.MapPost("/v1/images/generations", (ImageGenerationRequest request) =>
        {
            int seed = Random.Shared.Next();
            string outputFilename = Path.Combine(Path.GetTempPath(), $"{seed}.jpg");
            StableDiffusionBase sd = EnsureStableDiffusionInstance();
            sd.TextToImage(request.Prompt, outputFilename, seed);

            byte[] bytes = File.ReadAllBytes(outputFilename);
            string b64String = Convert.ToBase64String(bytes);
            app.Logger.LogInformation($"textToImage: {outputFilename}");

            ImageResponse response = new()
            {
                Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                Data = new UrlData[]
                {
            new UrlData()
            {
                Base64Json = b64String
            }
                }
            };

            return response;
        }).Accepts<ImageGenerationRequest>("application/json")
            .Produces<ImageResponse>(200);

        app.MapPost("/v1/images/edits", async (HttpRequest request) =>
        {
            IFormCollection form = await request.ReadFormAsync();
            IFormFile? image = form.Files.FirstOrDefault(file => file.Name == "image");
            IFormFile? mask = form.Files.FirstOrDefault(file => file.Name == "mask");
            string initImageFilename = string.Empty;
            string maskImageFilename = string.Empty;

            if (image != null && mask != null)
            {
                initImageFilename = WriteFormFile(image);
                maskImageFilename = WriteFormFile(mask);
            }
            // TODO - bad requests

            string prompt = form["prompt"].FirstOrDefault() ?? "";
            string? n = form["n"].FirstOrDefault();
            string? size = form["size"].FirstOrDefault();
            string? responseFormat = form["response_format"].FirstOrDefault();

            int seed = Random.Shared.Next();
            string outputFilename = Path.Combine(Path.GetTempPath(), $"{seed}.jpg");
            StableDiffusionBase sd = EnsureStableDiffusionInstance();
            maskImageFilename = sd.ConvertOpenAiMask(maskImageFilename);
            sd.Inpaint(prompt, initImageFilename, maskImageFilename, outputFilename, seed);

            byte[] bytes = File.ReadAllBytes(outputFilename);
            string b64String = Convert.ToBase64String(bytes);
            app.Logger.LogInformation($"inpaint: {outputFilename}");

            ImageResponse response = new()
            {
                Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                Data = new UrlData[]
                {
                    new UrlData()
                    {
                        Base64Json = b64String
                    }
                }
            };

            return response;
        }).Accepts<IFormFile>("multipart/form-data").Produces<ImageResponse>(200);

        app.MapPost("/v1/images/variations", async (HttpRequest request) =>
        {
            IFormCollection form = await request.ReadFormAsync();
            IFormFile? image = form.Files.FirstOrDefault(file => file.Name == "image");
            string initImageFilename = string.Empty;

            if (image != null)
            {
                initImageFilename = WriteFormFile(image);
            }
            // TODO - bad requests

            string prompt = form["prompt"].FirstOrDefault() ?? "";
            string? n = form["n"].FirstOrDefault();
            string? size = form["size"].FirstOrDefault();
            string? responseFormat = form["response_format"].FirstOrDefault();

            int seed = Random.Shared.Next();
            string outputFilename = Path.Combine(Path.GetTempPath(), $"{seed}.jpg");
            StableDiffusionBase sd = EnsureStableDiffusionInstance();
            sd.ImageToImage(prompt, initImageFilename, outputFilename, seed, 0.7f);

            byte[] bytes = File.ReadAllBytes(outputFilename);
            string b64String = Convert.ToBase64String(bytes);
            app.Logger.LogInformation($"imageToImage: {outputFilename}");

            ImageResponse response = new()
            {
                Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                Data = new UrlData[]
                {
                    new UrlData()
                    {
                        Base64Json = b64String
                    }
                }
            };

            return response;
        }).Accepts<IFormFile>("multipart/form-data")
            .Produces<ImageResponse>(200);

        app.Run();
    }

    private static string WriteFormFile(IFormFile file)
    {
        string outputPath = Path.GetTempFileName() + Path.GetExtension(file.FileName);
        using FileStream outputStream = File.Open(outputPath, FileMode.Create);
        using Stream sourceStream = file.OpenReadStream();
        sourceStream.CopyTo(outputStream);
        sourceStream.Close();
        outputStream.Close();

        return outputPath;
    }

    private static StableDiffusionBase EnsureStableDiffusionInstance()
    {
        if (KeepStableDiffusionRunning)
        {
            // use persistent instance
            if (_stableDiffusion == null)
            {
                _stableDiffusion = (StableDiffusionBase?)Activator.CreateInstance(SdModel,
                    @"C:\python\StableDiffusion", true);
                if (_stableDiffusion == null)
                {
                    throw new InvalidOperationException("failed to create Stable Diffusion instance");
                }
            }
            return _stableDiffusion;
        }
        else
        {
            // use 1 off instance that will exit after generation
            StableDiffusionBase? sd = (StableDiffusionBase?)Activator.CreateInstance(SdModel,
                @"C:\python\StableDiffusion", false);
            if (sd == null)
            {
                throw new InvalidOperationException("failed to create Stable Diffusion instance");
            }
            return sd;
        }
    }

    private static LlamaCpp EnsureLlamaInstance(List<ChatMessage>? contextMessages)
    {
        if (KeepLlamaRunning)
        {
            // use persistent instance
            if (_llama == null)
            {
                _llama = (LlamaCpp?)Activator.CreateInstance(LlmModel,
                    @"C:\llm\llama.cpp\bin\cuda12\main.exe",
                    @"C:\llm\llama.cpp\model");
                if (_llama == null)
                {
                    throw new InvalidOperationException("failed to create LLM instance");
                }
                _llama.StartSession(existingMessages: contextMessages, temp: Temperature);
            }
            return _llama;
        }
        else
        {
            LlamaCpp? llamaChat = (LlamaCpp?)Activator.CreateInstance(LlmModel,
                @"C:\llm\llama.cpp\bin\cuda12\main.exe",
                @"C:\llm\llama.cpp\model");
            if (_llama == null)
            {
                throw new InvalidOperationException("failed to create LLM instance");
            }
            llamaChat.StartSession(existingMessages: contextMessages);
            return llamaChat;
        }
    }
}