using System.Diagnostics;
using System.Reflection;

namespace StableDiffusionAdapter;

/// <summary>
/// This class directly interface with Stable Diffusion model and should fully expose its 
/// underlying features without having to be limited by any least common denominator API.
/// If we need to build some common interfaces (like OpenAI API), it would be built outside 
/// of this class.
/// 
/// The class has two modes, stateless and stateful mode:
///  * stateless - every call would load and unload pipeline
///  * stateful - calls would load the pipeline and it would stay in memory.
///    additional calls to the same pipeline would be faster.
///    subsequent call using a different pipeline would unload and load new pipeline
///    and then the new pipeline would stay in memory
/// </summary>
public abstract class StableDiffusionBase
{
    protected const string EndOfGenerationString = "GENERATION_DONE";
    protected string virtualEnvironmentRoot;
    protected string dllLocation;
    protected string i2iModelName;
    protected string t2iModelName;
    protected string inpaintModelName;
    protected bool keepPipelineRunning;
    protected Process? process = null;
    protected AutoResetEvent? generateCompleteEvent;

    /// <summary>
    /// Initialize variables only, no heavy work
    /// </summary>
    public StableDiffusionBase(
        string venvRoot, 
        string t2iModelName, 
        string inpaintModelName, 
        string i2iModelName,
        bool keepPipelineRunning)
    {
        this.t2iModelName = t2iModelName;
        this.i2iModelName = i2iModelName;
        this.inpaintModelName = inpaintModelName;
        this.virtualEnvironmentRoot = venvRoot;
        this.dllLocation = Path.GetDirectoryName(
            Assembly.GetExecutingAssembly().Location) ?? "";
        this.keepPipelineRunning = keepPipelineRunning;
        if (keepPipelineRunning)
        {
            this.generateCompleteEvent = new AutoResetEvent(initialState: false);
        }
    }

    public DiffusionPipelineType CurrentPipeline { get; set; } = 
        DiffusionPipelineType.None;

    public virtual void TextToImage(string prompt, string outputFilename, int seed = 0)
    {
        this.DoStableDiffusionWorkload(prompt, outputFilename, seed, 
            this.t2iModelName, "sd.py", DiffusionPipelineType.TextToImage, parameters: null);
    }

    public virtual void ImageToImage(
        string prompt, string initImage, string outputFilename, 
        int seed = 0, float strength = 0.5f)
    {
        this.EnsureExists(initImage);
        Dictionary<string, string> parameters = new()
        {
            { "--init-image", $"\"{initImage}\"" },
            { "--strength", $"{strength}" },
        };
        this.DoStableDiffusionWorkload(prompt, outputFilename, seed,
            this.i2iModelName, "sd_img2img.py", DiffusionPipelineType.ImageToImage, parameters);
    }

    public virtual void Inpaint(
        string prompt, string initImage, string mask, string outputFilename, 
        int seed = 0)
    {
        this.EnsureExists(initImage);
        this.EnsureExists(mask);
        Dictionary<string, string> parameters = new()
        {
            { "--init-image", $"\"{initImage}\"" },
            { "--mask", $"\"{mask}\"" },
        };
        this.DoStableDiffusionWorkload(prompt, outputFilename, seed,
            this.inpaintModelName, "sd_inpaint.py", DiffusionPipelineType.Inpaint, parameters);
    }

    public string ConvertOpenAiMask(string openAiMaskFile)
    {
        using Image<Rgba32> original = Image.Load<Rgba32>(openAiMaskFile);
        using Image<Rgba32> output = new Image<Rgba32>(
            original.Width, original.Height);

        for (int y = 0; y < original.Height; y++)
        {
            for (int x = 0; x < original.Width; x++)
            {
                if (original[x, y].A == 0)
                {
                    output[x, y] = new Rgba32(255, 255, 255, 255);
                }
                else
                {
                    output[x, y] = new Rgba32(0, 0, 0, 255);
                }
            }
        }

        string extention = Path.GetExtension(openAiMaskFile);
        string outputFile = openAiMaskFile.Replace(extention, "_sd" + extention);
        
        output.SaveAsPng(outputFile);
        return outputFile;
    }

    protected void DoStableDiffusionWorkload(
        string prompt, 
        string outputFilename,
        int seed,
        string modelName,
        string pythonScript,
        DiffusionPipelineType pipelineType,
        Dictionary<string, string>? parameters)
    {
        this.EnsureExists(Path.Combine(this.dllLocation, pythonScript));
        // escape and sanitize prompt
        prompt = this.EscapeString(prompt);
        string moreParameters = string.Empty;
        if (parameters != null)
        {
            moreParameters = string.Join(" ",
                parameters.Select(pair => $" {pair.Key} {pair.Value}").ToList());
        }

        if (keepPipelineRunning)
        {
            if (this.CurrentPipeline == pipelineType)
            {
                this.EnsureProcess();
                // signal running script to do work
                this.process!.StandardInput.WriteLine(prompt);
                this.process!.StandardInput.WriteLine(outputFilename);
                if (pipelineType == DiffusionPipelineType.Inpaint && parameters != null)
                {
                    // quote is not needed when passing over std in
                    this.process!.StandardInput.WriteLine(parameters["--init-image"].Trim('"'));
                    this.process!.StandardInput.WriteLine(parameters["--mask"].Trim('"'));
                } 
                else if (pipelineType == DiffusionPipelineType.ImageToImage && parameters != null)
                {
                    // quote is not needed when passing over std in
                    this.process!.StandardInput.WriteLine(parameters["--init-image"].Trim('"'));
                }
                this.generateCompleteEvent?.WaitOne();
            }
            else if (this.CurrentPipeline == DiffusionPipelineType.None)
            {
                // run the script in stateful mode
                string script =
                    $"{pythonScript} --model {modelName}" +
                    $" --prompt \"{prompt}\" --output \"{outputFilename}\" --seed {seed}" +
                    moreParameters + " --interactive";
                this.RunScript(script);
                this.CurrentPipeline = pipelineType;
            }
            else
            {
                //  shutdown the current script
                this.EnsureProcess();
                this.process!.Kill();
                this.process!.WaitForExit();
                // run the script again in stateful mode
                string script =
                    $"{pythonScript} --model {modelName}" +
                    $" --prompt \"{prompt}\" --output \"{outputFilename}\" --seed {seed}" +
                    moreParameters + " --interactive";
                this.RunScript(script);
                this.CurrentPipeline = pipelineType;
            }
        }
        else
        {
            // run the script in stateless mode
            string script =
                $"{pythonScript} --model {modelName}" +
                $" --prompt \"{prompt}\" --output \"{outputFilename}\" --seed {seed}" +
                moreParameters;
            this.RunScript(script);
        }
    }

    protected string EscapeString(string str)
    {
        string output = str.Replace("\"", "'");
        return output;
    }

    protected void EnsureExists(string filename)
    {
        if (!File.Exists(filename))
        {
            throw new ArgumentException($"{filename} does not exist");
        }
    }

    protected void EnsureProcess()
    {
        if (process == null)
        {
            // TODO: maybe relaunch to recover?
            throw new InvalidOperationException("SD script was not launched correctly");
        }

        if (process.HasExited)
        {
            // TODO: maybe relaunch to recover?
            throw new InvalidOperationException("SD script ended unexpectedly");
        }
    }

    protected void RunScript(string script)
    {
        Console.WriteLine(script);
        ProcessStartInfo psi = new()
        {
            FileName = Path.Combine(
                this.virtualEnvironmentRoot, @"Scripts\python.exe"),
            WorkingDirectory = this.dllLocation,
            Arguments = script,
            UseShellExecute = false,
            RedirectStandardError = true,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            CreateNoWindow = false,
        };
        psi.Environment.Add("HF_HOME", this.virtualEnvironmentRoot);

        this.process = Process.Start(psi);
        if (this.process == null)
        {
            throw new InvalidOperationException("launch SD script failed.");
        }

        this.process.OutputDataReceived += (sender, args) => {
            string line = args.Data ?? string.Empty;
            Console.WriteLine(line);
            if (line.Contains(EndOfGenerationString) && 
                this.generateCompleteEvent != null)
            {
                this.generateCompleteEvent.Set();
            }
        };
        this.process.ErrorDataReceived += (sender, args) => {
            Console.WriteLine(args.Data);
        };
        this.process.BeginOutputReadLine();
        this.process.BeginErrorReadLine();

        if (this.keepPipelineRunning)
        {
            this.generateCompleteEvent?.WaitOne();
            Console.WriteLine("generation completed");
        }
        else
        {
            this.process.WaitForExit();
            this.process.CancelErrorRead();
            this.process.CancelOutputRead();
            this.process = null;
            Console.WriteLine("process exited");
        }
    }
}

public enum DiffusionPipelineType
{
    None,
    TextToImage,
    ImageToImage,
    Inpaint
}