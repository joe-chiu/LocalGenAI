using OpenAiWebApi;
using System.Text;

namespace Llama2Adapter;

public abstract class Llama2ChatBase : LlamaCpp
{
    public Llama2ChatBase(string exePath, string modelPath) : base(
        exePath,
        modelPath,
        "[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n [/INST]",
        "You are a helpful and honest assistant. Always answer as helpfully as possible. " +
        "If a question does not make any sense, or is not factually coherent, " +
        "explain why instead of answering something not correct. "+ 
        "If you don't know the answer to a question, please don't share false information.")
    {
    }

    protected override string FormatPrompt(string system, string user, string assistant)
    {
        return $"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant} ";
    }

    protected override string FormatPrompt(string user, string assistant)
    {
        return $"[INST] {user} [/INST] {assistant} ";
    }
}
