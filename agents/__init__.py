def init_agent_from_config(config, device='cpu', normalization=None):
    """ Create an agent according to the config """

    agent_type = config.agent.type
    if agent_type == 'bc':
        from .BCAgent import _init_agent_from_config
        return _init_agent_from_config(config, device, normalization)

    if agent_type in ['bcimage', 'bcimagebyol']:
        from .BCImageAgent import _init_agent_from_config
        return _init_agent_from_config(config, device, normalization)

    if agent_type == 'bcimagegoal':
        from .BCImageGoalAgent import _init_agent_from_config
        return _init_agent_from_config(config, device, normalization)

    if agent_type == 'knn':
        from .KNN import _init_agent_from_config
        return _init_agent_from_config(config, normalization)

    if agent_type == 'morel':
        from .MORel import _init_agent_from_config
        return _init_agent_from_config(config, device)

    if agent_type == 'visuomotor':
        from .VisuoMotorPolicyWrapper import _init_agent_from_config
        return _init_agent_from_config(config, device)

    if agent_type == 'rb2':
        from .RB2PolicyWrapper import _init_agent_from_config
        return _init_agent_from_config(config, device)

    if agent_type == 'knn_jyo':
        from .knn_jyo import _init_agent_from_config
        return _init_agent_from_config(config, device)
    
    if agent_type == 'knn_byol':
        from .knn_byol import _init_agent_from_config
        return _init_agent_from_config(config, device)
    
    if agent_type == 'knn_image':
        from .knn_image import _init_agent_from_config
        return _init_agent_from_config(config, device)

    if agent_type == 'knn_audio_image':
        from .knn_audio_image import _init_agent_from_config
        return _init_agent_from_config(config, device)
    
    if agent_type == 'cql':
        from .CQL import _init_agent_from_config
        return _init_agent_from_config(config, device)
    
    if agent_type == 'd3rlpy':
        from .D3Agent import _init_agent_from_config
        return _init_agent_from_config(config, device)

    assert f"[ERROR] Unknown agent type {agent_type}"
