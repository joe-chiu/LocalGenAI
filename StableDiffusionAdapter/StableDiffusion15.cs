
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
public class StableDiffusion15 : StableDiffusionBase
{
    const string T2IModelName = "Lykon/dreamshaper-8";
    const string I2IModelName = "Lykon/dreamshaper-8";
    const string InpaintModelName = "Lykon/dreamshaper-8-inpainting";

    /// <summary>
    /// Initialize variables only, no heavy work
    /// </summary>
    public StableDiffusion15(string venvRoot, bool keepPipelineRunning = false) : 
        base(venvRoot, T2IModelName, InpaintModelName, I2IModelName, keepPipelineRunning)
    {
    }
}
