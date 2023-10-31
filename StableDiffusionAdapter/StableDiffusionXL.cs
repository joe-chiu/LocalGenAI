
// stage 1: every call is a self-contained run of a python script - stateless
// stage 2: manage a stateful python session, can more quickly run generation tasks

using System.Diagnostics;
using System.Reflection;

namespace StableDiffusionAdapter;

/// <summary>
/// This class directly interface with Stable Diffusion model and should fully expose its 
/// underlying features without having to be limited by least common denominator problem.
/// If we need to build some common interfaces, it would be built outside of this class.
/// </summary>
public class StableDiffusionXL : StableDiffusionBase
{
    const string ModelName = "stabilityai/stable-diffusion-xl-base-1.0";

    /// <summary>
    /// Initialize variables only, no heavy work
    /// </summary>
    public StableDiffusionXL(string venvRoot, bool keepPipelineRunning = false) : 
        base(venvRoot, ModelName, ModelName, ModelName, keepPipelineRunning)
    {
    }
}
