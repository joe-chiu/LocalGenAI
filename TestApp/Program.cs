
// for each adapter - test direct method interface and web service (OpenAI API)

using StableDiffusionAdapter;
using Llama2Adapter;
using System.Diagnostics;

internal class Program
{
    private static void Main(string[] args)
    {
        //TestStableDiffusionInteractive();
        //TestStableDiffusion(keepPipelineRunning: true);
        //TestStableDiffusionInteractive2();
        TestStableDiffusionInteractive3();
        //TestLlm();
    }

    private static void TestLlm()
    {
        LlamaCpp llamaChat = new Llama2Chat7B(
            @"C:\llm\llama.cpp\bin\cuda12\main.exe",
            @"C:\llm\llama.cpp\model");

        llamaChat.StartSession();
        string response1 = llamaChat.Chat("tell me about taiwan");
        llamaChat.EndSession();

        llamaChat.StartSession("You are an assistant who speaks like Donald Trump.");
        string response2 = llamaChat.Chat("tell me about taiwan");
        llamaChat.EndSession();
    }

    private static void TestStableDiffusionInteractive1()
    {
        StableDiffusionBase sd = new StableDiffusion15(
            @"C:\python\StableDiffusion", keepPipelineRunning: true);

        int seed = Random.Shared.Next();
        string filename = Path.Combine(Path.GetTempPath(), $"{seed}.jpg");

        sd.TextToImage("a magical computer computer displayed on a stand", filename, seed);
        ShowImage(filename);

        filename = Path.Combine(Path.GetTempPath(), $"{++seed}.jpg");
        sd.TextToImage("a beautiful landscape", filename, seed);
        ShowImage(filename);

        filename = Path.Combine(Path.GetTempPath(), $"{++seed}.jpg");
        sd.TextToImage("a beautiful lake on the moon", filename, seed);
        ShowImage(filename);
    }

    private static void TestStableDiffusionInteractive2()
    {
        StableDiffusionBase sd = new StableDiffusion15(
            @"C:\python\StableDiffusion", keepPipelineRunning: true);

        int seed = Random.Shared.Next();
        string filename = Path.Combine(Path.GetTempPath(), $"{seed}.jpg");

        sd.ImageToImage("a beutiful female Asian blacksmith", "blacksmith.png", filename, seed, 0.7f);
        ShowImage(filename);

        filename = Path.Combine(Path.GetTempPath(), $"{++seed}.jpg");
        sd.ImageToImage("an angry orc blacksmith", "blacksmith.png", filename, seed, 0.7f);
        ShowImage(filename);

        filename = Path.Combine(Path.GetTempPath(), $"{++seed}.jpg");
        sd.ImageToImage("a beutiful elf blacksmith", "blacksmith.png", filename, seed, 0.7f);
        ShowImage(filename);
    }

    private static void TestStableDiffusionInteractive3()
    {
        StableDiffusionBase sd = new StableDiffusion15(
            @"C:\python\StableDiffusion", keepPipelineRunning: true);

        int seed = Random.Shared.Next();
        string filename = Path.Combine(Path.GetTempPath(), $"{seed}.jpg");

        sd.Inpaint("a happy Asian blacksmith", "blacksmith.png", "blacksmith_mask.png", filename, seed);
        ShowImage(filename);

        filename = Path.Combine(Path.GetTempPath(), $"{++seed}.jpg");
        sd.Inpaint("a proud Asian blacksmith", "blacksmith.png", "blacksmith_mask.png", filename, seed);
        ShowImage(filename);

        filename = Path.Combine(Path.GetTempPath(), $"{++seed}.jpg");
        sd.Inpaint("a annoyed Asian blacksmith", "blacksmith.png", "blacksmith_mask.png", filename, seed);
        ShowImage(filename);
    }

    private static void TestStableDiffusion(bool keepPipelineRunning)
    {
        StableDiffusionBase sd = new StableDiffusion15(
            @"C:\python\StableDiffusion", keepPipelineRunning);

        int seed = Random.Shared.Next();
        string filename = Path.Combine(Path.GetTempPath(), $"{seed}.jpg");

        sd.TextToImage("a magical computer computer displayed on a stand", filename, seed);
        ShowImage(filename);

        sd.Inpaint("a happy Asian blacksmith", "blacksmith.png", "blacksmith_mask.png", filename, seed);
        ShowImage(filename);

        sd.ImageToImage("a beutiful female Asian blacksmith", "blacksmith.png", filename, seed, 0.7f);
        ShowImage(filename);
    }

    private static void ShowImage(string filename)
    {
        if (!File.Exists(filename))
        {
            return;
        }

        Process.Start(new ProcessStartInfo() { 
            UseShellExecute = true, 
            FileName = filename
        });
    }
}

