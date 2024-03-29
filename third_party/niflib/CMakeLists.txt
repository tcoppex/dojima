cmake_minimum_required(VERSION 3.10.2)

# -----------------------------------------------------------------------------

project(niflib)

option(NIFLIB_BUILD_SHARED_LIBS   ON)
option(NIFLIB_BUILD_TESTS         OFF)
option(NIFLIB_BUILD_DOCS          OFF)
option(NIFLIB_INSTALL             OFF)

file(STRINGS "VERSION" VERSION)

# -----------------------------------------------------------------------------

SET(sources

  src/RefObject.cpp
  src/kfm.cpp
  src/ComplexShape.cpp
  src/Inertia.cpp
  src/kfm.cpp
  src/MatTexCollection.cpp
  src/NIF_IO.cpp
  src/niflib.cpp
  src/nif_math.cpp
  src/RefObject.cpp
  src/Type.cpp

  # (emptied)
  src/nifqhull.cpp

  src/gen/AdditionalDataBlock.cpp
  src/gen/AdditionalDataInfo.cpp
  src/gen/ArkTexture.cpp
  src/gen/AVObject.cpp
  src/gen/bhkCMSDBigTris.cpp
  src/gen/bhkCMSDChunk.cpp
  # src/gen/bhkCMSDData.cpp
  src/gen/bhkCMSDMaterial.cpp
  src/gen/bhkCMSDBigTris.cpp
  src/gen/bhkCMSDTransform.cpp
  src/gen/BodyPartList.cpp
  src/gen/BoneLOD.cpp
  src/gen/BoundingBox.cpp
  src/gen/BoundingVolume.cpp
  src/gen/BoxBV.cpp
  src/gen/BSPackedAdditionalDataBlock.cpp
  src/gen/BSSegment.cpp
  src/gen/BSSegmentedTriangle.cpp
  src/gen/BSTreadTransfInfo.cpp
  src/gen/BSTreadTransform.cpp
  src/gen/BSTreadTransformData.cpp
  src/gen/BSTreadTransfSubInfo.cpp
  src/gen/ByteArray.cpp
  src/gen/ByteColor3.cpp
  src/gen/ByteColor4.cpp
  src/gen/ByteMatrix.cpp
  src/gen/CapsuleBV.cpp
  src/gen/ChannelData.cpp
  src/gen/ControllerLink.cpp
  src/gen/DecalVectorArray.cpp
  src/gen/ElementReference.cpp
  src/gen/enums.cpp
  src/gen/ExportInfo.cpp
  src/gen/ExtraMeshDataEpicMickey2.cpp
  src/gen/ExtraMeshDataEpicMickey.cpp
  src/gen/Footer.cpp
  src/gen/FurniturePosition.cpp
  src/gen/HalfSpaceBV.cpp
  src/gen/Header.cpp
  src/gen/HingeDescriptor.cpp
  src/gen/hkTriangle.cpp
  src/gen/LimitedHingeDescriptor.cpp
  src/gen/LODRange.cpp
  src/gen/MatchGroup.cpp
  src/gen/MaterialData.cpp
  src/gen/MeshData.cpp
  src/gen/MipMap.cpp
  src/gen/Morph.cpp
  src/gen/MorphWeight.cpp
  src/gen/MotorDescriptor.cpp
  src/gen/MTransform.cpp
  src/gen/MultiTextureElement.cpp
  src/gen/NodeGroup.cpp
  src/gen/OblivionColFilter.cpp
  src/gen/OblivionSubShape.cpp
  src/gen/OldSkinData.cpp
  src/gen/Particle.cpp
  src/gen/ParticleDesc.cpp
  src/gen/physXMaterialRef.cpp
  src/gen/Polygon.cpp
  src/gen/QTransform.cpp
  src/gen/QuaternionXYZW.cpp
  src/gen/RagdollDescriptor.cpp
  src/gen/Region.cpp
  src/gen/register.cpp
  src/gen/SemanticData.cpp
  src/gen/ShaderTexDesc.cpp
  src/gen/SkinData.cpp
  src/gen/SkinPartition.cpp
  src/gen/SkinPartitionUnknownItem1.cpp
  src/gen/SkinShape.cpp
  src/gen/SkinShapeGroup.cpp
  src/gen/SkinTransform.cpp
  src/gen/SkinWeight.cpp
  src/gen/SphereBV.cpp
  src/gen/StringPalette.cpp
  src/gen/TBC.cpp
  src/gen/TexDesc.cpp
  src/gen/TexSource.cpp
  src/gen/UnionBV.cpp
  src/gen/UnknownMatrix1.cpp

  src/obj/AbstractAdditionalGeometryData.cpp
  src/obj/ATextureRenderData.cpp
  src/obj/AvoidNode.cpp
  src/obj/bhkAabbPhantom.cpp
  src/obj/bhkBallAndSocketConstraint.cpp
  src/obj/bhkBallSocketConstraintChain.cpp
  src/obj/bhkBlendCollisionObject.cpp
  src/obj/bhkBlendController.cpp
  src/obj/bhkBoxShape.cpp
  src/obj/bhkBreakableConstraint.cpp
  src/obj/bhkBvTreeShape.cpp
  src/obj/bhkCapsuleShape.cpp
  src/obj/bhkCollisionObject.cpp
  src/obj/bhkCompressedMeshShape.cpp
  src/obj/bhkCompressedMeshShapeData.cpp
  src/obj/bhkConstraint.cpp
  src/obj/bhkConvexListShape.cpp
  src/obj/bhkConvexShape.cpp
  src/obj/bhkConvexTransformShape.cpp
  src/obj/bhkConvexVerticesShape.cpp
  src/obj/bhkEntity.cpp
  src/obj/bhkHingeConstraint.cpp
  src/obj/bhkLimitedHingeConstraint.cpp
  src/obj/bhkLiquidAction.cpp
  src/obj/bhkListShape.cpp
  src/obj/bhkMalleableConstraint.cpp
  src/obj/bhkMeshShape.cpp
  src/obj/bhkMoppBvTreeShape.cpp
  src/obj/bhkMultiSphereShape.cpp
  src/obj/bhkNiCollisionObject.cpp
  src/obj/bhkNiTriStripsShape.cpp
  src/obj/bhkOrientHingedBodyAction.cpp
  src/obj/bhkPackedNiTriStripsShape.cpp
  src/obj/bhkPCollisionObject.cpp
  src/obj/bhkPhantom.cpp
  src/obj/bhkPrismaticConstraint.cpp
  src/obj/bhkRagdollConstraint.cpp
  src/obj/bhkRefObject.cpp
  src/obj/bhkRigidBody.cpp
  src/obj/bhkRigidBodyT.cpp
  src/obj/bhkSerializable.cpp
  src/obj/bhkShapeCollection.cpp
  src/obj/bhkShape.cpp
  src/obj/bhkShapePhantom.cpp
  src/obj/bhkSimpleShapePhantom.cpp
  src/obj/bhkSPCollisionObject.cpp
  src/obj/bhkSphereRepShape.cpp
  src/obj/bhkSphereShape.cpp
  src/obj/bhkStiffSpringConstraint.cpp
  src/obj/bhkTransformShape.cpp
  src/obj/bhkWorldObject.cpp
  src/obj/BSAnimNotes.cpp
  src/obj/BSBehaviorGraphExtraData.cpp
  src/obj/BSBlastNode.cpp
  src/obj/BSBoneLODExtraData.cpp
  src/obj/BSBound.cpp
  src/obj/BSDamageStage.cpp
  src/obj/BSDebrisNode.cpp
  src/obj/BSDecalPlacementVectorExtraData.cpp
  src/obj/BSDismemberSkinInstance.cpp
  src/obj/BSDistantTreeShaderProperty.cpp
  src/obj/BSEffectShaderPropertyColorController.cpp
  src/obj/BSEffectShaderProperty.cpp
  src/obj/BSEffectShaderPropertyFloatController.cpp
  src/obj/BSFadeNode.cpp
  src/obj/BSFrustumFOVController.cpp
  src/obj/BSFurnitureMarker.cpp
  src/obj/BSFurnitureMarkerNode.cpp
  src/obj/BSInvMarker.cpp
  src/obj/BSKeyframeController.cpp
  src/obj/BSLagBoneController.cpp
  src/obj/BSLeafAnimNode.cpp
  src/obj/BSLightingShaderPropertyColorController.cpp
  src/obj/BSLightingShaderProperty.cpp
  src/obj/BSLightingShaderPropertyFloatController.cpp
  src/obj/BSLODTriShape.cpp
  src/obj/BSMasterParticleSystem.cpp
  src/obj/BSMaterialEmittanceMultController.cpp
  src/obj/BSMultiBoundAABB.cpp
  src/obj/BSMultiBound.cpp
  src/obj/BSMultiBoundData.cpp
  src/obj/BSMultiBoundNode.cpp
  src/obj/BSMultiBoundOBB.cpp
  src/obj/BSMultiBoundSphere.cpp
  src/obj/BSNiAlphaPropertyTestRefController.cpp
  src/obj/BSOrderedNode.cpp
  src/obj/BSPackedAdditionalGeometryData.cpp
  src/obj/BSParentVelocityModifier.cpp
  src/obj/BSProceduralLightningController.cpp
  src/obj/BSPSysArrayEmitter.cpp
  src/obj/BSPSysHavokUpdateModifier.cpp
  src/obj/BSPSysInheritVelocityModifier.cpp
  src/obj/BSPSysLODModifier.cpp
  src/obj/BSPSysMultiTargetEmitterCtlr.cpp
  src/obj/BSPSysRecycleBoundModifier.cpp
  src/obj/BSPSysScaleModifier.cpp
  src/obj/BSPSysSimpleColorModifier.cpp
  src/obj/BSPSysStripUpdateModifier.cpp
  src/obj/BSPSysSubTexModifier.cpp
  src/obj/BSRefractionFirePeriodController.cpp
  src/obj/BSRefractionStrengthController.cpp
  src/obj/BSRotAccumTransfInterpolator.cpp
  src/obj/BSSegmentedTriShape.cpp
  src/obj/BSShaderLightingProperty.cpp
  src/obj/BSShaderNoLightingProperty.cpp
  src/obj/BSShaderPPLightingProperty.cpp
  src/obj/BSShaderProperty.cpp
  src/obj/BSShaderTextureSet.cpp
  src/obj/BSSkyShaderProperty.cpp
  src/obj/BSStripParticleSystem.cpp
  src/obj/BSStripPSysData.cpp
  src/obj/BSTreadTransfInterpolator.cpp
  src/obj/BSTreeNode.cpp
  src/obj/BSValueNode.cpp
  src/obj/BSWArray.cpp
  src/obj/BSWaterShaderProperty.cpp
  src/obj/BSWindModifier.cpp
  src/obj/BSXFlags.cpp
  src/obj/CStreamableAssetData.cpp
  src/obj/DistantLODShaderProperty.cpp
  src/ObjectRegistry.cpp
  src/obj/FxButton.cpp
  src/obj/FxRadioButton.cpp
  src/obj/FxWidget.cpp
  src/obj/HairShaderProperty.cpp
  src/obj/hkPackedNiTriStripsData.cpp
  src/obj/Lighting30ShaderProperty.cpp
  src/obj/Ni3dsAlphaAnimator.cpp
  src/obj/Ni3dsAnimationNode.cpp
  src/obj/Ni3dsColorAnimator.cpp
  src/obj/Ni3dsMorphShape.cpp
  src/obj/Ni3dsParticleSystem.cpp
  src/obj/Ni3dsPathController.cpp
  src/obj/NiAdditionalGeometryData.cpp
  src/obj/NiAlphaController.cpp
  src/obj/NiAlphaProperty.cpp
  src/obj/NiAmbientLight.cpp
  src/obj/NiArkAnimationExtraData.cpp
  src/obj/NiArkImporterExtraData.cpp
  src/obj/NiArkShaderExtraData.cpp
  src/obj/NiArkTextureExtraData.cpp
  src/obj/NiArkViewportInfoExtraData.cpp
  src/obj/NiAutoNormalParticles.cpp
  src/obj/NiAutoNormalParticlesData.cpp
  src/obj/NiAVObject.cpp
  src/obj/NiAVObjectPalette.cpp
  src/obj/NiBezierMesh.cpp
  src/obj/NiBezierTriangle4.cpp
  src/obj/NiBillboardNode.cpp
  src/obj/NiBinaryExtraData.cpp
  src/obj/NiBinaryVoxelData.cpp
  src/obj/NiBinaryVoxelExtraData.cpp
  src/obj/NiBlendBoolInterpolator.cpp
  src/obj/NiBlendFloatInterpolator.cpp
  src/obj/NiBlendInterpolator.cpp
  src/obj/NiBlendPoint3Interpolator.cpp
  src/obj/NiBlendTransformInterpolator.cpp
  src/obj/NiBone.cpp
  src/obj/NiBoneLODController.cpp
  src/obj/NiBoolData.cpp
  src/obj/NiBooleanExtraData.cpp
  src/obj/NiBoolInterpController.cpp
  src/obj/NiBoolInterpolator.cpp
  src/obj/NiBoolTimelineInterpolator.cpp
  src/obj/NiBSAnimationNode.cpp
  src/obj/NiBSBoneLODController.cpp
  src/obj/NiBSPArrayController.cpp
  src/obj/NiBSParticleNode.cpp
  src/obj/NiBSplineBasisData.cpp
  src/obj/NiBSplineCompFloatInterpolator.cpp
  src/obj/NiBSplineCompPoint3Interpolator.cpp
  src/obj/NiBSplineCompTransformEvaluator.cpp
  src/obj/NiBSplineCompTransformInterpolator.cpp
  src/obj/NiBSplineData.cpp
  src/obj/NiBSplineFloatInterpolator.cpp
  src/obj/NiBSplineInterpolator.cpp
  src/obj/NiBSplinePoint3Interpolator.cpp
  src/obj/NiBSplineTransformInterpolator.cpp
  src/obj/NiCamera.cpp
  src/obj/NiClod.cpp
  src/obj/NiClodData.cpp
  src/obj/NiClodSkinInstance.cpp
  src/obj/NiCollisionData.cpp
  src/obj/NiCollisionObject.cpp
  src/obj/NiColorData.cpp
  src/obj/NiColorExtraData.cpp
  src/obj/NiControllerManager.cpp
  src/obj/NiControllerSequence.cpp
  src/obj/NiDataStream.cpp
  src/obj/NiDefaultAVObjectPalette.cpp
  src/obj/NiDirectionalLight.cpp
  src/obj/NiDitherProperty.cpp
  src/obj/NiDynamicEffect.cpp
  src/obj/NiEnvMappedTriShape.cpp
  src/obj/NiEnvMappedTriShapeData.cpp
  src/obj/NiExtraDataController.cpp
  src/obj/NiExtraData.cpp
  src/obj/NiFlipController.cpp
  src/obj/NiFloatData.cpp
  src/obj/NiFloatExtraDataController.cpp
  src/obj/NiFloatExtraData.cpp
  src/obj/NiFloatInterpController.cpp
  src/obj/NiFloatInterpolator.cpp
  src/obj/NiFloatsExtraData.cpp
  src/obj/NiFogProperty.cpp
  src/obj/NiFurSpringController.cpp
  src/obj/NiGeometry.cpp
  src/obj/NiGeometryData.cpp
  src/obj/NiGeomMorpherController.cpp
  src/obj/NiGravity.cpp
  src/obj/NiImage.cpp
  src/obj/NiInstancingMeshModifier.cpp
  src/obj/NiIntegerExtraData.cpp
  src/obj/NiIntegersExtraData.cpp
  src/obj/NiInterpController.cpp
  src/obj/NiInterpolator.cpp
  src/obj/NiKeyBasedInterpolator.cpp
  src/obj/NiKeyframeController.cpp
  src/obj/NiKeyframeData.cpp
  src/obj/NiLightColorController.cpp
  src/obj/NiLight.cpp
  src/obj/NiLightDimmerController.cpp
  src/obj/NiLightIntensityController.cpp
  src/obj/NiLines.cpp
  src/obj/NiLinesData.cpp
  src/obj/NiLODData.cpp
  src/obj/NiLODNode.cpp
  src/obj/NiLookAtController.cpp
  src/obj/NiLookAtInterpolator.cpp
  src/obj/NiMaterialColorController.cpp
  src/obj/NiMaterialProperty.cpp
  src/obj/NiMesh.cpp
  src/obj/NiMeshHWInstance.cpp
  src/obj/NiMeshModifier.cpp
  src/obj/NiMeshParticleSystem.cpp
  src/obj/NiMeshPSysData.cpp
  src/obj/NiMorphController.cpp
  src/obj/NiMorphData.cpp
  src/obj/NiMorpherController.cpp
  src/obj/NiMorphMeshModifier.cpp
  src/obj/NiMorphWeightsController.cpp
  src/obj/NiMultiTargetTransformController.cpp
  src/obj/NiMultiTextureProperty.cpp
  src/obj/NiNode.cpp
  src/obj/NiObject.cpp
  src/obj/NiObjectNET.cpp
  src/obj/NiPalette.cpp
  src/obj/NiParticleBomb.cpp
  src/obj/NiParticleColorModifier.cpp
  src/obj/NiParticleGrowFade.cpp
  src/obj/NiParticleMeshes.cpp
  src/obj/NiParticleMeshesData.cpp
  src/obj/NiParticleMeshModifier.cpp
  src/obj/NiParticleModifier.cpp
  src/obj/NiParticleRotation.cpp
  src/obj/NiParticles.cpp
  src/obj/NiParticlesData.cpp
  src/obj/NiParticleSystemController.cpp
  src/obj/NiParticleSystem.cpp
  src/obj/NiPathController.cpp
  src/obj/NiPathInterpolator.cpp
  src/obj/NiPersistentSrcTextureRendererData.cpp
  src/obj/NiPhysXActorDesc.cpp
  src/obj/NiPhysXBodyDesc.cpp
  src/obj/NiPhysXD6JointDesc.cpp
  src/obj/NiPhysXKinematicSrc.cpp
  src/obj/NiPhysXMaterialDesc.cpp
  src/obj/NiPhysXMeshDesc.cpp
  src/obj/NiPhysXProp.cpp
  src/obj/NiPhysXPropDesc.cpp
  src/obj/NiPhysXShapeDesc.cpp
  src/obj/NiPhysXTransformDest.cpp
  src/obj/NiPixelData.cpp
  src/obj/NiPlanarCollider.cpp
  src/obj/NiPoint3InterpController.cpp
  src/obj/NiPoint3Interpolator.cpp
  src/obj/NiPointLight.cpp
  src/obj/NiPortal.cpp
  src/obj/NiPosData.cpp
  src/obj/NiProperty.cpp
  src/obj/NiPSBombForce.cpp
  src/obj/NiPSBoundUpdater.cpp
  src/obj/NiPSBoxEmitter.cpp
  src/obj/NiPSCylinderEmitter.cpp
  src/obj/NiPSDragForce.cpp
  src/obj/NiPSEmitParticlesCtlr.cpp
  src/obj/NiPSEmitterDeclinationCtlr.cpp
  src/obj/NiPSEmitterDeclinationVarCtlr.cpp
  src/obj/NiPSEmitterLifeSpanCtlr.cpp
  src/obj/NiPSEmitterPlanarAngleCtlr.cpp
  src/obj/NiPSEmitterPlanarAngleVarCtlr.cpp
  src/obj/NiPSEmitterRadiusCtlr.cpp
  src/obj/NiPSEmitterRotAngleCtlr.cpp
  src/obj/NiPSEmitterRotAngleVarCtlr.cpp
  src/obj/NiPSEmitterRotSpeedCtlr.cpp
  src/obj/NiPSEmitterRotSpeedVarCtlr.cpp
  src/obj/NiPSEmitterSpeedCtlr.cpp
  src/obj/NiPSFacingQuadGenerator.cpp
  src/obj/NiPSForceActiveCtlr.cpp
  src/obj/NiPSGravityForce.cpp
  src/obj/NiPSGravityStrengthCtlr.cpp
  src/obj/NiPSMeshEmitter.cpp
  src/obj/NiPSMeshParticleSystem.cpp
  src/obj/NiPSParticleSystem.cpp
  src/obj/NiPSPlanarCollider.cpp
  src/obj/NiPSResetOnLoopCtlr.cpp
  src/obj/NiPSSimulatorCollidersStep.cpp
  src/obj/NiPSSimulator.cpp
  src/obj/NiPSSimulatorFinalStep.cpp
  src/obj/NiPSSimulatorForcesStep.cpp
  src/obj/NiPSSimulatorGeneralStep.cpp
  src/obj/NiPSSimulatorMeshAlignStep.cpp
  src/obj/NiPSSimulatorStep.cpp
  src/obj/NiPSSpawner.cpp
  src/obj/NiPSSphereEmitter.cpp
  src/obj/NiPSSphericalCollider.cpp
  src/obj/NiPSysAgeDeathModifier.cpp
  src/obj/NiPSysAirFieldAirFrictionCtlr.cpp
  src/obj/NiPSysAirFieldInheritVelocityCtlr.cpp
  src/obj/NiPSysAirFieldModifier.cpp
  src/obj/NiPSysAirFieldSpreadCtlr.cpp
  src/obj/NiPSysBombModifier.cpp
  src/obj/NiPSysBoundUpdateModifier.cpp
  src/obj/NiPSysBoxEmitter.cpp
  src/obj/NiPSysCollider.cpp
  src/obj/NiPSysColliderManager.cpp
  src/obj/NiPSysColorModifier.cpp
  src/obj/NiPSysCylinderEmitter.cpp
  src/obj/NiPSysData.cpp
  src/obj/NiPSysDragFieldModifier.cpp
  src/obj/NiPSysDragModifier.cpp
  src/obj/NiPSysEmitter.cpp
  src/obj/NiPSysEmitterCtlr.cpp
  src/obj/NiPSysEmitterCtlrData.cpp
  src/obj/NiPSysEmitterDeclinationCtlr.cpp
  src/obj/NiPSysEmitterDeclinationVarCtlr.cpp
  src/obj/NiPSysEmitterInitialRadiusCtlr.cpp
  src/obj/NiPSysEmitterLifeSpanCtlr.cpp
  src/obj/NiPSysEmitterPlanarAngleCtlr.cpp
  src/obj/NiPSysEmitterPlanarAngleVarCtlr.cpp
  src/obj/NiPSysEmitterSpeedCtlr.cpp
  src/obj/NiPSysFieldAttenuationCtlr.cpp
  src/obj/NiPSysFieldMagnitudeCtlr.cpp
  src/obj/NiPSysFieldMaxDistanceCtlr.cpp
  src/obj/NiPSysFieldModifier.cpp
  src/obj/NiPSysGravityFieldModifier.cpp
  src/obj/NiPSysGravityModifier.cpp
  src/obj/NiPSysGravityStrengthCtlr.cpp
  src/obj/NiPSysGrowFadeModifier.cpp
  src/obj/NiPSysInitialRotAngleCtlr.cpp
  src/obj/NiPSysInitialRotAngleVarCtlr.cpp
  src/obj/NiPSysInitialRotSpeedCtlr.cpp
  src/obj/NiPSysInitialRotSpeedVarCtlr.cpp
  src/obj/NiPSysMeshEmitter.cpp
  src/obj/NiPSysMeshUpdateModifier.cpp
  src/obj/NiPSysModifierActiveCtlr.cpp
  src/obj/NiPSysModifierBoolCtlr.cpp
  src/obj/NiPSysModifier.cpp
  src/obj/NiPSysModifierCtlr.cpp
  src/obj/NiPSysModifierFloatCtlr.cpp
  src/obj/NiPSysPlanarCollider.cpp
  src/obj/NiPSysPositionModifier.cpp
  src/obj/NiPSysRadialFieldModifier.cpp
  src/obj/NiPSysResetOnLoopCtlr.cpp
  src/obj/NiPSysRotationModifier.cpp
  src/obj/NiPSysSpawnModifier.cpp
  src/obj/NiPSysSphereEmitter.cpp
  src/obj/NiPSysSphericalCollider.cpp
  src/obj/NiPSysTrailEmitter.cpp
  src/obj/NiPSysTurbulenceFieldModifier.cpp
  src/obj/NiPSysUpdateCtlr.cpp
  src/obj/NiPSysVolumeEmitter.cpp
  src/obj/NiPSysVortexFieldModifier.cpp
  src/obj/NiRangeLODData.cpp
  src/obj/NiRawImageData.cpp
  src/obj/NiRenderObject.cpp
  src/obj/NiRollController.cpp
  src/obj/NiRoom.cpp
  src/obj/NiRoomGroup.cpp
  src/obj/NiRotatingParticles.cpp
  src/obj/NiRotatingParticlesData.cpp
  src/obj/NiScreenElements.cpp
  src/obj/NiScreenElementsData.cpp
  src/obj/NiScreenLODData.cpp
  src/obj/NiSequence.cpp
  src/obj/NiSequenceData.cpp
  src/obj/NiSequenceStreamHelper.cpp
  src/obj/NiShadeProperty.cpp
  src/obj/NiShadowGenerator.cpp
  src/obj/NiSingleInterpController.cpp
  src/obj/NiSkinData.cpp
  src/obj/NiSkinInstance.cpp
  src/obj/NiSkinningLODController.cpp
  src/obj/NiSkinningMeshModifier.cpp
  src/obj/NiSortAdjustNode.cpp
  src/obj/NiSourceCubeMap.cpp
  src/obj/NiSourceTexture.cpp
  src/obj/NiSpecularProperty.cpp
  src/obj/NiSphericalCollider.cpp
  src/obj/NiSpotLight.cpp
  src/obj/NiStencilProperty.cpp
  src/obj/NiStringExtraData.cpp
  src/obj/NiStringPalette.cpp
  src/obj/NiStringsExtraData.cpp
  src/obj/NiSwitchNode.cpp
  src/obj/NiTextKeyExtraData.cpp
  src/obj/NiTexture.cpp
  src/obj/NiTextureEffect.cpp
  src/obj/NiTextureModeProperty.cpp
  src/obj/NiTextureProperty.cpp
  src/obj/NiTextureTransformController.cpp
  src/obj/NiTexturingProperty.cpp
  src/obj/NiTimeController.cpp
  src/obj/NiTransformController.cpp
  src/obj/NiTransformData.cpp
  src/obj/NiTransformEvaluator.cpp
  src/obj/NiTransformInterpolator.cpp
  src/obj/NiTransparentProperty.cpp
  src/obj/NiTriBasedGeom.cpp
  src/obj/NiTriBasedGeomData.cpp
  src/obj/NiTriShape.cpp
  src/obj/NiTriShapeData.cpp
  src/obj/NiTriShapeSkinController.cpp
  src/obj/NiUVController.cpp
  src/obj/NiUVData.cpp
  src/obj/NiVectorExtraData.cpp
  src/obj/NiVertexColorProperty.cpp
  src/obj/NiVertWeightsExtraData.cpp
  src/obj/NiVisController.cpp
  src/obj/NiVisData.cpp
  src/obj/NiWireframeProperty.cpp
  src/obj/NiZBufferProperty.cpp
  src/obj/RootCollisionNode.cpp
  src/obj/SkyShaderProperty.cpp
  src/obj/TallGrassShaderProperty.cpp
  src/obj/TileShaderProperty.cpp
  src/obj/VolumetricFogShaderProperty.cpp
  src/obj/WaterShaderProperty.cpp

  src/obj/NiSkinPartition.cpp
  src/obj/NiTriStrips.cpp
  src/obj/NiTriStripsData.cpp

  TriStripper/connectivity_graph.cpp
  TriStripper/policy.cpp
  TriStripper/tri_stripper.cpp

  NvTriStrip/NvTriStrip.cpp
  NvTriStrip/NvTriStripObjects.cpp
  NvTriStrip/VertexCache.cpp
)

# -----------------------------------------------------------------------------

include_directories("include/")


if(NIFLIB_BUILD_SHARED_LIBS)
  set(PROJECT_LIB_TYPE SHARED)
else()
  set(PROJECT_LIB_TYPE STATIC)
endif()

set(TARGET_NAME nif)
add_library(${TARGET_NAME} ${PROJECT_LIB_TYPE} ${sources})

# set_target_properties(niflib
#   PROPERTIES DEFINE_SYMBOL BUILDING_NIFLIB_DLL
# )
# add_library(niflib_static STATIC ${sources})
# set_target_properties(niflib_static
#    PROPERTIES
#    COMPILE_DEFINITIONS NIFLIB_STATIC_LINK
# )

#find boost and enable testing
if (NIFLIB_BUILD_TESTS)
  find_package(Boost 1.75.0 EXACT)
  if(Boost_FOUND)
  	INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
  	ADD_DEFINITIONS("-DHAS_BOOST")
  	enable_testing()
  	add_subdirectory(test)
  endif()
endif()

# build documentation
if(NIFLIB_BUILD_DOCS)
  find_package(Doxygen)
  if(DOXYGEN_FOUND)
    SET(DOXYGEN_INPUT ${CMAKE_BINARY_DIR}/Doxyfile)
    SET(DOXYGEN_OUTPUT ${CMAKE_BINARY_DIR}/doc/html/index.html)
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
      ${DOXYGEN_INPUT}
      @ONLY
    )
    add_custom_command(
      OUTPUT ${DOXYGEN_OUTPUT}
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_INPUT}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
    add_custom_target(apidocs ALL DEPENDS ${DOXYGEN_OUTPUT})
    set_directory_properties(PROPERTIES
      ADDITIONAL_MAKE_CLEAN_FILES doc
    )
    install(
      DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc/html/
      DESTINATION share/doc/${PROJECT_NAME}/apidocs
    )
  endif(DOXYGEN_FOUND)
endif()

# install
if (NIFLIB_INSTALL)
  install(
    TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )

  install(
    DIRECTORY include/
    DESTINATION include
  )

  install(
    FILES license.txt VERSION # change_log.txt
    DESTINATION share/doc/${PROJECT_NAME}
  )
endif()

# # enable CPack
# set(CPACK_PACKAGE_VERSION ${VERSION})
# set(CPACK_GENERATOR ZIP)
# if(MSVC)
#   include(CMakeDetermineVSServicePack)
#   DetermineVSServicePack(VCTAG)
#   if(VCTAG)
#     if(CMAKE_CL_64)
#       set(CPACK_SYSTEM_NAME win64-${VCTAG})
#     else()
#       set(CPACK_SYSTEM_NAME win32-${VCTAG})
#     endif()
#   endif()
# endif()
# set(CPACK_SOURCE_PACKAGE_FILE_NAME "niflib-${VERSION}-src")
# set(CPACK_SOURCE_GENERATOR ZIP)
# set(CPACK_SOURCE_IGNORE_FILES
#     "~$"
#     "/\\\\.git"
#     "/\\\\.project$"
#     "/\\\\.gitmodules$"
#     "/\\\\.gitattributes$"
#     "/\\\\.gitignore$"
# )
# include(CPack)
