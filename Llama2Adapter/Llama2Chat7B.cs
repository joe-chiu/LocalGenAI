
using OpenAiWebApi;

namespace Llama2Adapter
{
    public class Llama2Chat7B : Llama2ChatBase
    {
        public Llama2Chat7B(string exePath, string modelFolder) : base(
            exePath,
            Path.Combine(modelFolder, "llama-2-7b-chat.Q4_K_M.gguf"))
        {
        }
    }
}
