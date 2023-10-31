
using OpenAiWebApi;

namespace Llama2Adapter
{
    public class MythomaxL213B : LlamaCpp
    {
        public MythomaxL213B(string exePath, string modelFolder) : base(
            exePath,
            Path.Combine(modelFolder, "mythomax-l2-13b.Q5_K_M.gguf"),
            // --instruct would default to Alpaca prompt format
            "{system_prompt}\n\n", "")
        {
        }
    }
}
