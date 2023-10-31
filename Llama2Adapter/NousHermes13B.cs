
using OpenAiWebApi;
using System.Text;

namespace Llama2Adapter
{
    public class NousHermes13B : LlamaCpp
    {
        public NousHermes13B(string exePath, string modelFolder) : base(
            exePath,
            Path.Combine(modelFolder, "nous-hermes-llama2-13b.Q5_K_M.gguf"),
            // --instruct would default to Alpaca prompt format
            "{system_prompt}\n\n", "")
        {
        }
    }
}
