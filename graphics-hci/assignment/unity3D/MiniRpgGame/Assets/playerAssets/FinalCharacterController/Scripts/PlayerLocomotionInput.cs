using Unity.Cinemachine;
using UnityEngine;
using UnityEngine.InputSystem;

namespace playerAssets.FinalCharacterController
{
    [DefaultExecutionOrder(-2)]
    public class PlayerLocomotionInput : MonoBehaviour, PlayerControls.IPlayerLocomotionMapActions
    {
        #region Class Variables
        [SerializeField] private bool holdToSprint = true;

        public bool SprintToggledOn { get; private set; }
        public PlayerControls PlayerControls { get; private set; }
        public Vector2 MovementInput { get; private set; }
        public Vector2 LookInput { get; private set; }

        public bool JumpPressed { get; private set; }
        #endregion


        public PlayerController Controller { get; private set; }

        #region Game Objects
        [Header("Camera")]
        public Camera firstPersonCamera;
        public Camera thirdPersonCamera;
        public Camera dummyCamera;
        public CinemachineThirdPersonAim virtualCamera;

        [Header("AudioListner")]
        public AudioListener firstPersonAudio;
        public AudioListener thirdPersonAudio;
        #endregion

        private void OnEnable()
        {
            PlayerControls = new PlayerControls();
            Controller = GetComponent<PlayerController>();
            PlayerControls.Enable();

            PlayerControls.PlayerLocomotionMap.Enable();
            PlayerControls.PlayerLocomotionMap.SetCallbacks(this);
        }

        private void OnDisable()
        {
            PlayerControls.PlayerLocomotionMap.Disable();
            PlayerControls.PlayerLocomotionMap.RemoveCallbacks(this);
        }

        public void OnMovement(InputAction.CallbackContext context)
        {
            MovementInput = context.ReadValue<Vector2>();
           
        }

        public void OnLook(InputAction.CallbackContext context)
        {
            LookInput = context.ReadValue<Vector2>();
        }

        public void OnSwitch(InputAction.CallbackContext context)
        {
            if (context.performed)
            {
                firstPersonCamera.enabled = !firstPersonCamera.enabled;
                virtualCamera.enabled = !virtualCamera.enabled;
                firstPersonAudio.enabled = !firstPersonAudio.enabled;
                thirdPersonAudio.enabled = !thirdPersonAudio.enabled;
                if (firstPersonAudio.enabled)
                {
                    Controller.SetActiveCamera(firstPersonCamera);
                }
                else
                {
                    Controller.SetActiveCamera(dummyCamera);
                }
            }
        }

        public void OnToggleSprint(InputAction.CallbackContext context)
        {
            if (context.performed)
            {
                SprintToggledOn = holdToSprint || !SprintToggledOn;

            }
            else if (context.canceled)
            {
                SprintToggledOn = !holdToSprint && SprintToggledOn;
            }
        }
        private void LateUpdate()
        {
            JumpPressed = false;
        }

        public void OnJump(InputAction.CallbackContext context)
        {
            if(!context.performed)
            {
                return;
            }
            JumpPressed = true;
        }

        public void OnAttack(InputAction.CallbackContext context)
        {
            Debug.Log("Gracz atakuje");
        }
    }

}

