using OpenAiWebApi;
using System.Diagnostics;
using System.Text;

namespace Llama2Adapter;

public abstract class LlamaCpp
{
    protected string exePath;
    protected string modelPath;
    protected string promptTemplate;
    protected Process? process;
    protected StringBuilder screenBuffer;
    protected string defaultSystemMessage;
    protected AutoResetEvent generationCompleteEvent;
    protected const string EndOfGeneration = "> ";

    /// <summary>
    /// Base class for all LLM models that llama.cpp can inference
    /// </summary>
    /// <param name="exePath">Path to main.exe for llama.cpp</param>
    /// <param name="modelPath">Path to model file (.gguf)</param>
    /// <param name="promptTemplate">The prompt to initiailize the model. Can contain system messages or
    /// prior few-shot conversations as example or context. Response to this prompt is not generated</param>
    /// <param name="defaultSystemMessage">Fallback if user did not specify a system message.
    /// System message is useful for setting up AI functionality like role playing target
    /// or other generation parameters like respond in rhyme.</param>
    public LlamaCpp(string exePath, string modelPath, string promptTemplate, string defaultSystemMessage)
    {
        this.exePath = exePath;
        this.modelPath = modelPath;
        this.promptTemplate = promptTemplate;
        this.screenBuffer = new StringBuilder();
        this.defaultSystemMessage = defaultSystemMessage;
        this.generationCompleteEvent = new AutoResetEvent(initialState: false);
    }

    /// <summary>
    /// Start a new chat session.
    /// Should not use both systemMessage and existingMessages at the same time.
    /// existingMessages can contain system message already.
    /// </summary>
    /// <param name="systemMessage">Provide a simple way to set system message</param>
    /// <param name="topK"></param>
    /// <param name="temp"></param>
    /// <param name="repeatPenalty"></param>
    /// <exception cref="InvalidOperationException"></exception>
    public void StartSession(
        string? systemMessage = null,
        float temp = 0.8f,
        float repeatPenalty = 1.1f)
    {
        StartSession(systemMessage, null, temp, repeatPenalty);
    }

    /// <summary>
    /// Start a new chat session.
    /// Should not use both systemMessage and existingMessages at the same time.
    /// existingMessages can contain system message already.
    /// </summary>
    /// <param name="existingMessages">For setting up more complex multishot conversation</param>
    /// <param name="topK"></param>
    /// <param name="temp"></param>
    /// <param name="repeatPenalty"></param>
    /// <exception cref="InvalidOperationException"></exception>
    public void StartSession(
        List<ChatMessage>? existingMessages,
        float temp = 0.8f,
        float repeatPenalty = 1.1f)
    {
        StartSession(null, existingMessages, temp, repeatPenalty);
    }

    /// <summary>
    /// Start a new chat session.
    /// Should not use both systemMessage and existingMessages at the same time.
    /// existingMessages can contain system message already.
    /// </summary>
    /// <param name="systemMessage">Provide a simple way to set system message</param>
    /// <param name="existingMessages">For setting up more complex multishot conversation</param>
    /// <param name="temp"></param>
    /// <param name="repeatPenalty"></param>
    /// <exception cref="InvalidOperationException"></exception>
    protected void StartSession(
        string? systemMessage, List<ChatMessage>? existingMessages, float temp, float repeatPenalty)
    {
        this.screenBuffer.Clear();

        string initPrompt = this.FormatInitialPrompt(systemMessage, existingMessages);

        // the \n prefix would force the > prompt to be printed and we can use that as
        // generation complete indicator
        string arguments = $"--model {this.modelPath} --temp {temp} --repeat_penalty {repeatPenalty} " +
            "--ctx_size 0 --instruct --log-disable --n-gpu-layers 100 --seed -1 --in-prefix \"\n\" " + 
            $"--prompt \"{initPrompt}\"";

        RunExe(arguments);
        if (this.process == null)
        {
            throw new InvalidOperationException("llama.cpp not running");
        }

        // llama.cpp print "> " when ready for user input in "instruct mode"
        this.generationCompleteEvent.WaitOne();
    }

    public void EndSession()
    {
        if (this.process == null)
        {
            throw new InvalidOperationException("llama.cpp not running");
        }

        this.process.CancelErrorRead();
        this.process.CancelOutputRead();
        this.process.Kill();
        this.process.WaitForExit();
        this.process = null;
    }

    public string Chat(string message)
    {
        if (this.process == null)
        {
            throw new InvalidOperationException("llama.cpp not running");
        }

        int startLength = this.screenBuffer.Length;
        this.generationCompleteEvent.Reset();
        // by default, the input is not echoed, so print them manually to Console
        // NOT in screen buffer, unless that is needed
        Console.WriteLine(message);
        this.process.StandardInput.WriteLine(message);
        this.generationCompleteEvent.WaitOne();
        string screen = screenBuffer.ToString();
        string response = screen.Substring(
            // "> \n" - remove the trailing 3 characters
            startLength, this.screenBuffer.Length - startLength - 3);
        // remove junk strings that sometimes come from the model
        response = this.CleanResponse(response);
        return response;
    }

    protected virtual string FormatPrompt(string system, string user, string assistant)
    {
        return $"{system}\n\n### Instruction:\n\n{user}\n\n### Response:\n\n{assistant}\n\n";
    }

    protected virtual string FormatPrompt(string user, string assistant)
    {
        return $"### Instruction:\n\n{user}\n\n### Response:\n\n{assistant}\n\n";
    }

    /// <summary>
    /// Formulate initial prompt for various situations:
    /// 1. nothing - no initial prompt
    /// 2. system message only from <paramref name="systemMessage"/>
    /// 3. system message only from <paramref name="existingMessages"/>
    /// 4. system message + N * (user prompt, assistant) pairs from <paramref name="existingMessages"/>
    /// Other permutations are possible but won't support unless there is use cases
    /// </summary>
    /// <param name="systemMessage"></param>
    /// <param name="existingMessages"></param>
    /// <returns></returns>
    protected string FormatInitialPrompt(string? systemMessage, List<ChatMessage>? existingMessages)
    {
        if (existingMessages == null)
        {
            return this.promptTemplate
                .Replace("{system_prompt}", systemMessage ?? this.defaultSystemMessage);
        }

        existingMessages = existingMessages ?? new();

        string[] systemMessages = existingMessages
            .Where(message => message.Role == ChatRole.System)
            .Select(message => message.Content)
            .ToArray();
        string joinedSystemMessage = string.Join(" ", systemMessages);

        ChatMessage[] conversations = existingMessages
            .Where(message => message.Role == ChatRole.Assistant || message.Role == ChatRole.User)
            .ToArray();

        if (conversations.Length == 0)
        {
            return this.promptTemplate
                .Replace("{system_prompt}", joinedSystemMessage);
        }

        StringBuilder sb = new();
        for (int i = 0; i < conversations.Length; i++)
        {
            if (i + 1 < conversations.Length &&
                conversations[i].Role == ChatRole.User &&
                conversations[i + 1].Role == ChatRole.Assistant)
            {
                if (i == 0 && !string.IsNullOrWhiteSpace(joinedSystemMessage))
                {
                    // first one contains system message
                    sb.Append(this.FormatPrompt(joinedSystemMessage, 
                        conversations[i].Content, conversations[i + 1].Content));
                }
                else
                {
                    sb.Append(this.FormatPrompt(
                        conversations[i].Content, conversations[i + 1].Content));
                }

                // process pair at a time
                i++;
            }
        }

        return sb.ToString();
    }

    protected string CleanResponse(string input)
    {
        string[] stringsToRemove =
        {
            // observe a lot of these showing up in response
            "### Instruction:",
        };

        foreach(string str in stringsToRemove)
        {
            input = input.Replace(str, "");
        }

        return input.Trim();
    }

    protected void RunExe(string arguments)
    {
        Console.WriteLine(arguments);
        string exeDir = Path.GetDirectoryName(this.exePath) ?? string.Empty;
        ProcessStartInfo psi = new()
        {
            FileName = this.exePath,
            WorkingDirectory = exeDir,
            Arguments = arguments,
            UseShellExecute = false,
            RedirectStandardError = true,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            CreateNoWindow = false,
        };

        Process? proc = Process.Start(psi);
        if (proc == null)
        {
            return;
        }

        this.process = proc;
        proc.OutputDataReceived += (sender, args) => {
            Console.WriteLine(args.Data);
            this.screenBuffer.Append(args.Data);
            this.screenBuffer.Append("\n");
            if (args.Data == EndOfGeneration)
            {
                this.generationCompleteEvent.Set();
            }
        };
        proc.ErrorDataReceived += (sender, args) => {
            // these are debug info from llama.cpp, not needed
            /*
            Console.WriteLine(args.Data);
            this.screenBuffer.Append(args.Data);
            this.screenBuffer.Append("\n");
            */
        };
        proc.BeginOutputReadLine();
        proc.BeginErrorReadLine();
    }

    // this would be needed if we can't rely on llama.cpp "instruct mode"
    // if only in "interactive mode", llama.cpp would not print "> " prompt
    // to indicate generation is complete
    protected bool WaitForText(string target)
    {
        bool hasMatch = false;
        TimeSpan timeout = TimeSpan.FromMinutes(3);
        DateTimeOffset startTime = DateTimeOffset.Now;

        List<int> matches = new();

        while (!hasMatch)
        {
            foreach (ReadOnlyMemory<char> chunk in this.screenBuffer.GetChunks())
            {
                for (int j = 0; j < chunk.Span.Length; j++)
                {
                    // bingo card algorithm, everyone gets the same bingo card -> target string
                    // every time we see a match on thr first letter, we start a new player
                    // each character read from enumerator, every player check their bingo card until they win
                    // win = all bingo card match (target string match)

                    // each player tracks a position in the target string (starting from 0)

                    char c = chunk.Span[j];

                    if (c == target[0])
                    {
                        // first character match, start a new bingo card
                        matches.Add(0);
                    }

                    // index to remove would be stored from largest index to smallest
                    // so it can be removed correctly
                    Stack<int> indexToRemove = new Stack<int>();
                    for (int i = 0; i < matches.Count; i++)
                    {
                        if (matches[i] == target.Length)
                        {
                            // bingo!
                            return true;
                        }

                        if (target[matches[i]] != c)
                        {
                            // failed match, will remove
                            indexToRemove.Push(i);
                        }

                        matches[i]++;
                    }

                    while (indexToRemove.Count > 0)
                    {
                        int i = indexToRemove.Pop();
                        matches.RemoveAt(i);
                    }
                }
            }

            if (DateTimeOffset.Now - startTime > timeout)
            {
                break;
            }

            if (hasMatch)
            {
                break;
            }
            else
            {
                Thread.Sleep(100);
            }
        }

        return hasMatch;
    }
}
