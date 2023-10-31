
namespace Llama2Adapter
{
    public class Llama2Chat13B : Llama2ChatBase
    {
        public Llama2Chat13B(string exePath, string modelFolder) : base(
            exePath,
            Path.Combine(modelFolder, "llama-2-13b-chat.Q5_K_M.gguf"))
        {
        }
    }
}
