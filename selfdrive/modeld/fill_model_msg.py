import os
import capnp
import numpy as np
from cereal import log
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan, Meta

SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')

ConfidenceClass = log.ModelDataV2.ConfidenceClass


class PublishState:
  def __init__(self):
    self.disengage_buffer = np.zeros(ModelConstants.CONFIDENCE_BUFFER_LEN*ModelConstants.DISENGAGE_WIDTH, dtype=np.float32)
    self.prev_brake_5ms2_probs = np.zeros(ModelConstants.FCW_5MS2_PROBS_WIDTH, dtype=np.float32)
    self.prev_brake_3ms2_probs = np.zeros(ModelConstants.FCW_3MS2_PROBS_WIDTH, dtype=np.float32)

def fill_xyzt(builder, t, x, y, z, x_std=None, y_std=None, z_std=None):
  builder.t = t
  builder.x = x
  builder.y = y
  builder.z = z
  if x_std is not None:
    builder.xStd = x_std
  if y_std is not None:
    builder.yStd = y_std
  if z_std is not None:
    builder.zStd = z_std

def fill_xyvat(builder, t, x, y, v, a, x_std=None, y_std=None, v_std=None, a_std=None):
  builder.t = t
  builder.x = x
  builder.y = y
  builder.v = v
  builder.a = a
  if x_std is not None:
    builder.xStd = x_std
  if y_std is not None:
    builder.yStd = y_std
  if v_std is not None:
    builder.vStd = v_std
  if a_std is not None:
    builder.aStd = a_std

def fill_xyz_poly(builder, degree, x, y, z):
  xyz = np.column_stack([x, y, z])
  coeffs = np.polynomial.polynomial.polyfit(ModelConstants.T_IDXS, xyz, deg=degree)
  c = coeffs.T.tolist()
  builder.xCoefficients = c[0]
  builder.yCoefficients = c[1]
  builder.zCoefficients = c[2]

def fill_lane_line_meta(builder, lane_lines, lane_line_probs):
  builder.leftY = lane_lines[1].y[0]
  builder.leftProb = lane_line_probs[1]
  builder.rightY = lane_lines[2].y[0]
  builder.rightProb = lane_line_probs[2]

def fill_model_msg(base_msg: capnp._DynamicStructBuilder, extended_msg: capnp._DynamicStructBuilder,
                   net_output_data: dict[str, np.ndarray], action: log.ModelDataV2.Action,
                   publish_state: PublishState, vipc_frame_id: int, vipc_frame_id_extra: int,
                   frame_id: int, frame_drop: float, timestamp_eof: int, model_execution_time: float,
                   valid: bool) -> None:
  frame_age = frame_id - vipc_frame_id if frame_id > vipc_frame_id else 0
  frame_drop_perc = frame_drop * 100
  extended_msg.valid = valid
  base_msg.valid = valid

  driving_model_data = base_msg.drivingModelData

  driving_model_data.frameId = vipc_frame_id
  driving_model_data.frameIdExtra = vipc_frame_id_extra
  driving_model_data.frameDropPerc = frame_drop_perc
  driving_model_data.modelExecutionTime = model_execution_time

  driving_model_data.action = action

  modelV2 = extended_msg.modelV2
  modelV2.frameId = vipc_frame_id
  modelV2.frameIdExtra = vipc_frame_id_extra
  modelV2.frameAge = frame_age
  modelV2.frameDropPerc = frame_drop_perc
  modelV2.timestampEof = timestamp_eof
  modelV2.modelExecutionTime = model_execution_time

  # plan — batch-convert 2D arrays to Python lists (fewer .tolist() calls)
  plan = net_output_data['plan'][0]
  plan_stds = net_output_data['plan_stds'][0]
  pos = plan[:,Plan.POSITION].T.tolist()
  pos_std = plan_stds[:,Plan.POSITION].T.tolist()
  vel = plan[:,Plan.VELOCITY].T.tolist()
  acc = plan[:,Plan.ACCELERATION].T.tolist()
  ori = plan[:,Plan.T_FROM_CURRENT_EULER].T.tolist()
  ori_rate = plan[:,Plan.ORIENTATION_RATE].T.tolist()

  fill_xyzt(modelV2.position, ModelConstants.T_IDXS, *pos, *pos_std)
  fill_xyzt(modelV2.velocity, ModelConstants.T_IDXS, *vel)
  fill_xyzt(modelV2.acceleration, ModelConstants.T_IDXS, *acc)
  fill_xyzt(modelV2.orientation, ModelConstants.T_IDXS, *ori)
  fill_xyzt(modelV2.orientationRate, ModelConstants.T_IDXS, *ori_rate)

  # poly path
  fill_xyz_poly(driving_model_data.path, ModelConstants.POLY_PATH_DEGREE, *pos)

  # action
  modelV2.action = action

  # times at X_IDXS of edges and lines aren't used
  LINE_T_IDXS: list[float] = []
  x_idxs_list = list(ModelConstants.X_IDXS)

  # lane lines — batch-convert per lane
  modelV2.init('laneLines', 4)
  ll_data = net_output_data['lane_lines'][0]
  for i in range(4):
    ll_yz = ll_data[i].T.tolist()
    fill_xyzt(modelV2.laneLines[i], LINE_T_IDXS, x_idxs_list, *ll_yz)
  modelV2.laneLineStds = net_output_data['lane_lines_stds'][0,:,0,0].tolist()
  modelV2.laneLineProbs = net_output_data['lane_lines_prob'][0,1::2].tolist()

  fill_lane_line_meta(driving_model_data.laneLineMeta, modelV2.laneLines, modelV2.laneLineProbs)

  # road edges — batch-convert per edge
  modelV2.init('roadEdges', 2)
  re_data = net_output_data['road_edges'][0]
  for i in range(2):
    re_yz = re_data[i].T.tolist()
    fill_xyzt(modelV2.roadEdges[i], LINE_T_IDXS, x_idxs_list, *re_yz)
  modelV2.roadEdgeStds = net_output_data['road_edges_stds'][0,:,0,0].tolist()

  # leads — batch-convert per lead
  modelV2.init('leadsV3', 3)
  for i in range(3):
    lead = modelV2.leadsV3[i]
    lead_vals = net_output_data['lead'][0,i].T.tolist()
    lead_stds = net_output_data['lead_stds'][0,i].T.tolist()
    fill_xyvat(lead, ModelConstants.LEAD_T_IDXS, *lead_vals, *lead_stds)
    lead.prob = net_output_data['lead_prob'][0,i].tolist()
    lead.probTime = ModelConstants.LEAD_T_OFFSETS[i]

  # meta — bulk convert meta array once
  meta = modelV2.meta
  meta.desireState = net_output_data['desire_state'][0].reshape(-1).tolist()
  meta.desirePrediction = net_output_data['desire_pred'][0].reshape(-1).tolist()
  meta.engagedProb = net_output_data['meta'][0,Meta.ENGAGED].item()
  meta.init('disengagePredictions')
  disengage_predictions = meta.disengagePredictions
  disengage_predictions.t = ModelConstants.META_T_IDXS
  meta_data = net_output_data['meta'][0]
  disengage_predictions.brakeDisengageProbs = meta_data[Meta.BRAKE_DISENGAGE].tolist()
  disengage_predictions.gasDisengageProbs = meta_data[Meta.GAS_DISENGAGE].tolist()
  disengage_predictions.steerOverrideProbs = meta_data[Meta.STEER_OVERRIDE].tolist()
  disengage_predictions.brake3MetersPerSecondSquaredProbs = meta_data[Meta.HARD_BRAKE_3].tolist()
  disengage_predictions.brake4MetersPerSecondSquaredProbs = meta_data[Meta.HARD_BRAKE_4].tolist()
  disengage_predictions.brake5MetersPerSecondSquaredProbs = meta_data[Meta.HARD_BRAKE_5].tolist()
  disengage_predictions.gasPressProbs = meta_data[Meta.GAS_PRESS].tolist()
  disengage_predictions.brakePressProbs = meta_data[Meta.BRAKE_PRESS].tolist()

  publish_state.prev_brake_5ms2_probs[:-1] = publish_state.prev_brake_5ms2_probs[1:]
  publish_state.prev_brake_5ms2_probs[-1] = net_output_data['meta'][0,Meta.HARD_BRAKE_5][0]
  publish_state.prev_brake_3ms2_probs[:-1] = publish_state.prev_brake_3ms2_probs[1:]
  publish_state.prev_brake_3ms2_probs[-1] = net_output_data['meta'][0,Meta.HARD_BRAKE_3][0]
  hard_brake_predicted = (publish_state.prev_brake_5ms2_probs > ModelConstants.FCW_THRESHOLDS_5MS2).all() and \
    (publish_state.prev_brake_3ms2_probs > ModelConstants.FCW_THRESHOLDS_3MS2).all()
  meta.hardBrakePredicted = hard_brake_predicted.item()

  # confidence
  if vipc_frame_id % (2*ModelConstants.MODEL_RUN_FREQ) == 0:
    # any disengage prob
    brake_disengage_probs = net_output_data['meta'][0,Meta.BRAKE_DISENGAGE]
    gas_disengage_probs = net_output_data['meta'][0,Meta.GAS_DISENGAGE]
    steer_override_probs = net_output_data['meta'][0,Meta.STEER_OVERRIDE]
    any_disengage_probs = 1-((1-brake_disengage_probs)*(1-gas_disengage_probs)*(1-steer_override_probs))
    # independent disengage prob for each 2s slice
    ind_disengage_probs = np.r_[any_disengage_probs[0], np.diff(any_disengage_probs) / (1 - any_disengage_probs[:-1])]
    # rolling buf for 2, 4, 6, 8, 10s
    publish_state.disengage_buffer[:-ModelConstants.DISENGAGE_WIDTH] = publish_state.disengage_buffer[ModelConstants.DISENGAGE_WIDTH:]
    publish_state.disengage_buffer[-ModelConstants.DISENGAGE_WIDTH:] = ind_disengage_probs

  score = 0.
  for i in range(ModelConstants.DISENGAGE_WIDTH):
    score += publish_state.disengage_buffer[i*ModelConstants.DISENGAGE_WIDTH+ModelConstants.DISENGAGE_WIDTH-1-i].item() / ModelConstants.DISENGAGE_WIDTH
  if score < ModelConstants.RYG_GREEN:
    modelV2.confidence = ConfidenceClass.green
  elif score < ModelConstants.RYG_YELLOW:
    modelV2.confidence = ConfidenceClass.yellow
  else:
    modelV2.confidence = ConfidenceClass.red

  # raw prediction if enabled
  if SEND_RAW_PRED:
    modelV2.rawPredictions = net_output_data['raw_pred'].tobytes()

def fill_pose_msg(msg: capnp._DynamicStructBuilder, net_output_data: dict[str, np.ndarray],
                  vipc_frame_id: int, vipc_dropped_frames: int, timestamp_eof: int, live_calib_seen: bool) -> None:
  msg.valid = live_calib_seen & (vipc_dropped_frames < 1)
  cameraOdometry = msg.cameraOdometry

  cameraOdometry.frameId = vipc_frame_id
  cameraOdometry.timestampEof = timestamp_eof

  pose = net_output_data['pose'][0].tolist()
  pose_stds = net_output_data['pose_stds'][0].tolist()
  cameraOdometry.trans = pose[:3]
  cameraOdometry.rot = pose[3:]
  cameraOdometry.wideFromDeviceEuler = net_output_data['wide_from_device_euler'][0,:].tolist()
  cameraOdometry.roadTransformTrans = net_output_data['road_transform'][0,:3].tolist()
  cameraOdometry.transStd = pose_stds[:3]
  cameraOdometry.rotStd = pose_stds[3:]
  cameraOdometry.wideFromDeviceEulerStd = net_output_data['wide_from_device_euler_stds'][0,:].tolist()
  cameraOdometry.roadTransformTransStd = net_output_data['road_transform_stds'][0,:3].tolist()
