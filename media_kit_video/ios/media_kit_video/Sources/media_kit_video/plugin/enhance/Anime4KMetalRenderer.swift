//  Part of media_kit_video (Kazumi fork). See LICENSE.
//
//  Metal port of the Anime4K shader chain used by the Kazumi iOS video
//  enhancement path. MPVShader turns an mpv-style .glsl shader pack into Metal
//  compute kernels; this renderer builds one pipeline per enabled shader, runs
//  them over an rgba16Float texture chain and finishes with a bilinear blit
//  into the caller's render target (the plugin's BGRA pixel buffer texture).
//
//  Copyright 2021 Yi Xie
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not
//  use this file except in compliance with the License. You may obtain a copy of
//  the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
//  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
//  License for the specific language governing permissions and limitations under
//  the License.
//
//  Ported from https://github.com/imxieyi/Anime4KMetal (Apache-2.0).

import Foundation
import Metal

final class Anime4KMetalRenderer {

    /// Rotating texture slots: record the next frame while the GPU reads the old one.
    static let bufferCount = 2

    let name: String
    let device: MTLDevice
    let shaders: [MPVShader]
    let libraries: [MTLLibrary]

    var enabledShaders: [MPVShader]
    var pipelineStates: [MTLComputePipelineState]

    /// Bilinear blit into the drawable-sized target (inline kernel, no .metal file).
    let finalizePipelineState: MTLComputePipelineState
    let finalizeSamplerState: MTLSamplerState

    /// Per buffer slot. "MAIN"/"NATIVE" alias the input, "output" is the chain
    /// result, every other key is a SAVE target allocated on demand.
    var textureMap: [[String: MTLTexture]]
    var nearestSamplerStates: [MTLSamplerState]
    var linearSamplerStates: [MTLSamplerState]

    /// Size of every named texture as seen by WHEN expressions. OUTPUT must be
    /// the pixel size of the Flutter video area: AutoDownscalePre depends on it.
    var sizeMap: [String: (Float, Float)]
    var bufferIndex = -1

    var outputW: Float = 0
    var outputH: Float = 0
    var textureInW: Float = 0
    var textureInH: Float = 0

    private static let finalizeKernelSource = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void anime4k_finalize(texture2d<float, access::sample> src [[texture(0)]],
                                 texture2d<float, access::write> dst [[texture(1)]],
                                 sampler s [[sampler(0)]],
                                 uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) { return; }
        float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
        dst.write(src.sample(s, uv), gid);
    }
    """

    init(name: String, device: MTLDevice, glslSources: [String]) throws {
        self.name = name
        self.device = device
        self.enabledShaders = []
        self.pipelineStates = []
        self.textureMap = []
        self.nearestSamplerStates = []
        self.linearSamplerStates = []
        self.sizeMap = [:]

        var parsed: [MPVShader] = []
        for source in glslSources {
            parsed.append(contentsOf: try MPVShader.parse(source))
        }
        guard !parsed.isEmpty else { throw Anime4KMetalError.encoderFail("No shader parsed for '\(name)'") }
        self.shaders = parsed
        // Compilation errors from makeLibrary(source:options:) are surfaced as is.
        let metalDevice = device // keep the closure off self before full init
        self.libraries = try parsed.map { shader in
            try metalDevice.makeLibrary(source: shader.metalCode, options: nil)
        }
        print("[Anime4K] '\(name)': \(parsed.count) shader(s) -> " + parsed.map { $0.name }.joined(separator: ", "))

        let samplerDesc = MTLSamplerDescriptor()
        samplerDesc.magFilter = .linear
        samplerDesc.minFilter = .linear
        samplerDesc.sAddressMode = .clampToEdge
        samplerDesc.tAddressMode = .clampToEdge
        guard let finalizeSampler = device.makeSamplerState(descriptor: samplerDesc) else { throw Anime4KMetalError.encoderFail("Failed to create finalize sampler") }
        self.finalizeSamplerState = finalizeSampler

        let finalizeLibrary = try device.makeLibrary(source: Anime4KMetalRenderer.finalizeKernelSource, options: nil)
        guard let finalizeFunction = finalizeLibrary.makeFunction(name: "anime4k_finalize") else { throw Anime4KMetalError.functionNotFound("anime4k_finalize") }
        self.finalizePipelineState = try device.makeComputePipelineState(function: finalizeFunction)
    }

    // MARK: - Pipeline build

    /// Builds the pipeline list for one frame size. Call again whenever the
    /// input size or the video area size changes.
    func compileShaders(
        device: MTLDevice,
        textureInW: Int,
        textureInH: Int,
        outputW: Int,
        outputH: Int
    ) throws {
        enabledShaders.removeAll()
        nearestSamplerStates.removeAll()
        linearSamplerStates.removeAll()
        pipelineStates.removeAll()
        textureMap.removeAll()
        sizeMap.removeAll()
        bufferIndex = -1

        self.textureInW = Float(textureInW)
        self.textureInH = Float(textureInH)
        self.outputW = self.textureInW
        self.outputH = self.textureInH
        sizeMap["MAIN"] = (Float(textureInW), Float(textureInH))
        sizeMap["NATIVE"] = (Float(textureInW), Float(textureInH))
        sizeMap["OUTPUT"] = (Float(outputW), Float(outputH))

        let binaryOperators: [Substring] = ["+", "-", "*", "/", "<", ">"]

        for i in 0..<shaders.count {
            let shader = shaders[i]

            // Evaluate the WHEN condition using Reverse Polish notation.
            if let when = shader.when {
                let splits = when.split(separator: " ").compactMap { item -> Substring? in
                    if item == "WHEN" || item.isEmpty {
                        return nil
                    }
                    return item
                }
                var stack: [Float] = []
                for token in splits {
                    let tSplits = token.split(separator: ".")
                    if tSplits.count == 2 && (tSplits[1] == "w" || tSplits[1] == "h") {
                        let sizeName = String(tSplits[0])
                        guard let size = sizeMap[sizeName] else { throw Anime4KMetalError.encoderFail("Unknown size '\(sizeName)' in WHEN condition: \(when)") }
                        stack.append(tSplits[1] == "w" ? size.0 : size.1)
                        continue
                    }
                    if binaryOperators.contains(token) {
                        guard stack.count >= 2 else { throw Anime4KMetalError.encoderFail("Malformed WHEN condition: \(when)") }
                        let rhs = stack.removeLast()
                        let lhs = stack.removeLast()
                        switch token {
                        case "+":
                            stack.append(lhs + rhs)
                        case "-":
                            stack.append(lhs - rhs)
                        case "*":
                            stack.append(lhs * rhs)
                        case "/":
                            stack.append(lhs / rhs)
                        case "<":
                            stack.append(lhs < rhs ? 1 : 0)
                        case ">":
                            stack.append(lhs > rhs ? 1 : 0)
                        default:
                            throw Anime4KMetalError.encoderFail("Unknown operator '\(token)' in WHEN condition: \(when)")
                        }
                        continue
                    }
                    guard let value = Float(token) else { throw Anime4KMetalError.encoderFail("Malformed token '\(token)' in WHEN condition: \(when)") }
                    stack.append(value)
                }
                guard stack.count == 1 else { throw Anime4KMetalError.encoderFail("Failed to evaluate WHEN condition: \(when)") }
                if stack.removeLast() == 0 {
                    print("[Anime4K] Skip shader \(shader.name)")
                    continue
                }
            }

            enabledShaders.append(shader)
            let library = libraries[i]

            self.outputW = self.textureInW
            self.outputH = self.textureInH
            if let hooked = shader.hook {
                sizeMap["HOOKED"] = sizeMap[hooked]
            }
            if let widthMultiplier = shader.width {
                guard let base = sizeMap[widthMultiplier.0] else { throw Anime4KMetalError.encoderFail("Shader \(shader.name) multiplies unknown width '\(widthMultiplier.0)'") }
                self.outputW = base.0 * widthMultiplier.1
            }
            if let heightMultiplier = shader.height {
                guard let base = sizeMap[heightMultiplier.0] else { throw Anime4KMetalError.encoderFail("Shader \(shader.name) multiplies unknown height '\(heightMultiplier.0)'") }
                self.outputH = base.1 * heightMultiplier.1
            }
            if let save = shader.save, save != "MAIN" {
                sizeMap[save] = (self.outputW, self.outputH)
            }
            guard let function = library.makeFunction(name: shader.functionName) else { throw Anime4KMetalError.functionNotFound(shader.functionName) }
            pipelineStates.append(try device.makeComputePipelineState(function: function))
        }
    }

    // MARK: - Encoding

    /// Records the whole chain into the command buffer and returns the final
    /// rgba16Float texture. Returns the input unchanged when no shader is enabled.
    func encode(commandBuffer: MTLCommandBuffer, input: MTLTexture) throws -> MTLTexture {
        guard pipelineStates.count == enabledShaders.count else { throw Anime4KMetalError.encoderFail("Pipeline state count \(pipelineStates.count) mismatch shader count \(enabledShaders.count)") }
        if enabledShaders.isEmpty {
            return input
        }

        bufferIndex = (bufferIndex + 1) % Anime4KMetalRenderer.bufferCount
        if textureMap.count <= bufferIndex {
            textureMap.append([:])
            let desc = MTLSamplerDescriptor()
            desc.magFilter = .nearest
            desc.minFilter = .nearest
            desc.sAddressMode = .clampToEdge
            desc.tAddressMode = .clampToEdge
            guard let nearestSampler = device.makeSamplerState(descriptor: desc) else { throw Anime4KMetalError.encoderFail("Failed to create nearest sampler") }
            nearestSamplerStates.append(nearestSampler)
            desc.magFilter = .linear
            desc.minFilter = .linear
            guard let linearSampler = device.makeSamplerState(descriptor: desc) else { throw Anime4KMetalError.encoderFail("Failed to create linear sampler") }
            linearSamplerStates.append(linearSampler)
        }
        guard nearestSamplerStates.count > bufferIndex && linearSamplerStates.count > bufferIndex else { throw Anime4KMetalError.encoderFail("Sampler states missing for buffer slot \(bufferIndex)") }

        textureMap[bufferIndex]["MAIN"] = input
        textureMap[bufferIndex]["NATIVE"] = input

        let desc = MTLTextureDescriptor()
        desc.width = Int(outputW)
        desc.height = Int(outputH)
        desc.pixelFormat = .rgba16Float
        desc.usage = [.shaderWrite, .shaderRead]
        desc.storageMode = .private
        guard let chainOutput = device.makeTexture(descriptor: desc) else { throw Anime4KMetalError.encoderFail("Failed to allocate chain output texture") }
        textureMap[bufferIndex]["output"] = chainOutput

        for i in 0..<enabledShaders.count {
            let shader = enabledShaders[i]
            var outputW = textureInW
            var outputH = textureInH
            if let hooked = shader.hook {
                sizeMap["HOOKED"] = sizeMap[hooked]
            }
            if let widthMultiplier = shader.width {
                guard let base = sizeMap[widthMultiplier.0] else { throw Anime4KMetalError.encoderFail("Shader \(shader.name) multiplies unknown width '\(widthMultiplier.0)'") }
                outputW = base.0 * widthMultiplier.1
            }
            if let heightMultiplier = shader.height {
                guard let base = sizeMap[heightMultiplier.0] else { throw Anime4KMetalError.encoderFail("Shader \(shader.name) multiplies unknown height '\(heightMultiplier.0)'") }
                outputH = base.1 * heightMultiplier.1
            }
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw Anime4KMetalError.encoderCreationFail(shader.functionName) }
            let pipelineState = pipelineStates[i]
            encoder.setComputePipelineState(pipelineState)
            if outputW >= textureInW {
                encoder.setSamplerState(nearestSamplerStates[bufferIndex], index: 0)
            } else {
                encoder.setSamplerState(linearSamplerStates[bufferIndex], index: 0)
            }
            for j in 0..<shader.inputTextureNames.count {
                var textureName = shader.inputTextureNames[j]
                if textureName == "HOOKED", let hook = shader.hook {
                    textureName = hook
                }
                if !textureMap[bufferIndex].keys.contains(textureName) {
                    if textureName == shader.save {
                        let desc = MTLTextureDescriptor()
                        desc.width = Int(outputW)
                        desc.height = Int(outputH)
                        desc.pixelFormat = .rgba16Float
                        desc.usage = [.shaderWrite, .shaderRead]
                        desc.storageMode = .private
                        guard let texture = device.makeTexture(descriptor: desc) else { throw Anime4KMetalError.encoderFail("Failed to allocate texture \(textureName)") }
                        textureMap[bufferIndex][textureName] = texture
                    } else {
                        throw Anime4KMetalError.encoderFail("texture \(textureName) is missing")
                    }
                }
                encoder.setTexture(textureMap[bufferIndex][textureName], index: j)
            }
            if shader.binds.contains(shader.outputTextureName)
                || !textureMap[bufferIndex].keys.contains(shader.outputTextureName) {
                let desc = MTLTextureDescriptor()
                desc.width = Int(outputW)
                desc.height = Int(outputH)
                desc.pixelFormat = .rgba16Float
                desc.usage = [.shaderWrite, .shaderRead]
                desc.storageMode = .private
                guard let texture = device.makeTexture(descriptor: desc) else { throw Anime4KMetalError.encoderFail("Failed to allocate output texture \(shader.outputTextureName)") }
                textureMap[bufferIndex][shader.outputTextureName] = texture
            }
            guard let outputTex = textureMap[bufferIndex][shader.outputTextureName] else { throw Anime4KMetalError.encoderFail("output texture \(shader.outputTextureName) is missing") }
            encoder.setTexture(outputTex, index: shader.inputTextureNames.count)
            let w = pipelineState.threadExecutionWidth
            let h = pipelineState.maxTotalThreadsPerThreadgroup / w
            let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
            let threadgroupsPerGrid = MTLSize(width: (outputTex.width + w - 1) / w,
                                              height: (outputTex.height + h - 1) / h,
                                              depth: outputTex.arrayLength)
            encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }

        guard let result = textureMap[bufferIndex]["output"] else { throw Anime4KMetalError.encoderFail("Chain output texture is missing") }
        return result
    }

    /// Runs the chain and blits the result into the given drawable-sized texture.
    func encode(commandBuffer: MTLCommandBuffer, input: MTLTexture, output: MTLTexture) throws {
        let chainOutput = try encode(commandBuffer: commandBuffer, input: input)
        let w = finalizePipelineState.threadExecutionWidth
        let h = finalizePipelineState.maxTotalThreadsPerThreadgroup / w
        guard w > 0, h > 0 else { throw Anime4KMetalError.encoderFail("Invalid threadgroup size for anime4k_finalize") }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw Anime4KMetalError.encoderCreationFail("anime4k_finalize") }
        encoder.setComputePipelineState(finalizePipelineState)
        encoder.setTexture(chainOutput, index: 0)
        encoder.setTexture(output, index: 1)
        encoder.setSamplerState(finalizeSamplerState, index: 0)
        let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
        let threadgroupsPerGrid = MTLSize(width: (output.width + w - 1) / w,
                                          height: (output.height + h - 1) / h,
                                          depth: output.arrayLength)
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}

enum Anime4KMetalError: Error, LocalizedError {
    case encoderCreationFail(String)
    case encoderFail(String)
    case functionNotFound(String)

    var errorDescription: String? {
        switch self {
        case let .encoderCreationFail(msg):
            return "Cannot create encoder for " + msg
        case let .encoderFail(msg):
            return "Failed to encode: " + msg
        case let .functionNotFound(msg):
            return "Cannot find Metal function: " + msg
        }
    }
}