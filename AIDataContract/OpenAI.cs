using System.Runtime.Serialization;
using System.Text.Json.Serialization;

namespace OpenAiWebApi;

public class ImageGenerationRequest
{
    public ImageGenerationRequest()
    {
        this.Count = 1;
        this.Size = ImageSize.Square1024;
        this.ResponseFormat = ImageResponseFormat.Base64Json;
    }

    public required string Prompt { get; set; }

    [JsonPropertyName("n")]
    public int Count { get; set; }

    public ImageSize Size { get; set; }

    [JsonPropertyName("response_format")]
    public ImageResponseFormat ResponseFormat { get; set; }
}

public class ImageResponse
{
    public required long Created { get; set; }
    public required UrlData[] Data { get; set; }
}

public class UrlData
{
    public string? Url { get; set; }

    [JsonPropertyName("b64_json")]
    public string? Base64Json { get; set; }
}

public enum ImageSize
{
    [EnumMember(Value = "256x256")]
    Square256,
    [EnumMember(Value = "512x512")]
    Square512,
    [EnumMember(Value = "1024x1024")]
    Square1024
}
public enum ImageResponseFormat
{
    Url,
    [EnumMember(Value = "b64_json")]
    Base64Json
}

public class ChatCreateRequest
{
    public required string Model { get; set; }

    public required ChatMessage[] Messages { get; set; }

    public bool Stream { get; set; }

    public float Temperature { get; set; }

    [JsonPropertyName("max_tokens")]
    public int MaxTokens { get; set; }

    [JsonPropertyName("top_p")]
    public float NucleusSamplingFactor { get; set; }

    [JsonPropertyName("frequency_penalty")]
    public int FrequencyPenalty { get; set; }

    [JsonPropertyName("presence_penalty")]
    public int PresencePenalty { get; set; }
}


public class ChatCompletionResponse
{
    public required string Id { get; set; }
    public required string Object { get; set; }
    public required long Created { get; set; }
    public required string Model { get; set; }
    public required ChatCompletionChoice[] Choices { get; set; }
}

public class ChatCompletionChunk
{
    public required string Id { get; set; }

    public required string Model { get; set; }

    public required ChatCompletionDelta[] Choices { get; set; }
}

public enum ChatRole
{
    System,
    User,
    Assistant
}

public class ChatMessage
{
    public ChatRole Role { get; set; }

    public string Content { get; set; } = string.Empty;
}

public enum FinishReason
{
    Stop,
    Length,
    [EnumMember(Value = "function_call")]
    FunctionCall,
    [EnumMember(Value = "content_filter")]
    ContentFilter
}

public class ChatCompletionChoice
{
    public int Index { get; set; }

    public required ChatMessage Message { get; set; }

    [JsonPropertyName("finish_reason")]
    public FinishReason? FinishReason { get; set; }
}

public class ChatCompletionDelta
{
    public int Index { get; set; }

    public required ChatMessage Delta { get; set; }

    [JsonPropertyName("finish_reason")]
    public FinishReason? FinishReason { get; set; }
}