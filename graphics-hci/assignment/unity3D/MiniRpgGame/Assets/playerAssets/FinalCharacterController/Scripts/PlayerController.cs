using System;
using UnityEngine;


namespace playerAssets.FinalCharacterController
{
    [DefaultExecutionOrder(-1)]
    public class PlayerController : MonoBehaviour
    {
        #region Class Variables
        [Header("Components*")]
        [SerializeField] private CharacterController _characterController;
        [SerializeField] private Camera _playerCamera;

        [Header("Movement Settings*")]
        public float runAcceleration = 0.25f;
        public float runSpeed = 4f;
        public float sprintAcceleration = 0.5f;
        public float sprintSpeed = 7f;
        public float inAirAcceleration = 25f;
        public float drag = 0.1f;
        public float movingThreshold = 0.01f;
        public float gravity = 25f;
        public float jumpSpeed = 1.0f;

        [Header("Camer Settings*")]
        public float lookSenseH = 0.1f;
        public float lookSenseV = 0.1f;
        public float lookLimitV = 89f;

        [Header("Environment Details")]
        [SerializeField] private LayerMask _groundLayers;

        [Header("Attack settings*")]
        public float attackRange = 2f;
        public int attackDamage = 25;
        public float attackCooldown = 0.8f;
        private bool canAttack = true;
        public LayerMask enemyLayers;

        private PlayerLocomotionInput _playerLocomotionInput;
        private PlayerState _playerState;
        private Vector2 _cameraRotation = Vector2.zero;
        private Vector2 _playerTargetRotation = Vector2.zero;

        private float _verticalVelocity = 0f;
        private float _antiBymp;
        private bool _jumpedLastFrame = false;
        private float _stepOffset;
        #endregion


        public void SetActiveCamera(Camera newCam)
        {
            _playerCamera = newCam;
        }

        void Start()
        {
            Cursor.lockState = CursorLockMode.Locked;
            
        }
        private void Awake()
        {
            _playerLocomotionInput = GetComponent<PlayerLocomotionInput>();
            _playerState = GetComponent<PlayerState>();

            _antiBymp = sprintSpeed;
            _stepOffset = _characterController.stepOffset;
        }

        private void Update()
        {
            UpdateMovementState();
            HandleVerticalMovement();
            HandleLateralMovement();
            HandleAttack();
        }
        private void HandleAttack()
        {
            if (_playerLocomotionInput.AttackPressed && canAttack)
            {
                canAttack = false;
                Vector3 attackOrigin = transform.position + transform.forward * 1.0f;

                // Szukamy kolider�w tylko na warstwie "Enemy"
                Collider[] hits = Physics.OverlapSphere(attackOrigin, attackRange, enemyLayers);

                foreach (Collider hit in hits)
                {
                    if (hit.TryGetComponent<EnemyAiTutorial>(out EnemyAiTutorial enemy))
                    {
                        enemy.TakeDamage(attackDamage);
                        
                        Debug.Log($"Trafiono {hit.name} i zadano {attackDamage} obra�e�.");
                    }
                }
                Invoke(nameof(ResetAttack), attackCooldown);
            }
        }
        private void ResetAttack()
        {
            canAttack = true;
        }


        private void UpdateMovementState()
        {
            bool isMovementInput = _playerLocomotionInput.MovementInput != Vector2.zero;
            bool isMovingLaterally = IsMovingLaterally();
            bool isSprinting = _playerLocomotionInput.SprintToggledOn && isMovingLaterally;
            bool isGrounded = IsGrounded();

            PlayerMovementState lateralState = isSprinting ? PlayerMovementState.Sprinting :
                                               isMovingLaterally || isMovingLaterally ? PlayerMovementState.Running : PlayerMovementState.Idling;
            _playerState.SetPlayerMovementState(lateralState);

            if((!isGrounded || _jumpedLastFrame) && _characterController.velocity.y >= 0f)
            {
                _playerState.SetPlayerMovementState(PlayerMovementState.Jumping);
                _jumpedLastFrame = false;
                _characterController.stepOffset = 0f;
            }
            else if ((!isGrounded || _jumpedLastFrame) && _characterController.velocity.y < 0f)
            {
                _playerState.SetPlayerMovementState(PlayerMovementState.Falling);
                _jumpedLastFrame = false;
                _characterController.stepOffset = 0f;
            }
            else
            {
                _characterController.stepOffset = _stepOffset;
            }
        }

        private void HandleVerticalMovement()
        {
            bool isGrounded = _playerState.InGroundedState();
            _verticalVelocity -= gravity * Time.deltaTime;

            if (isGrounded && _verticalVelocity < 0)
            {
                _verticalVelocity = -_antiBymp;
            }
            if (_playerLocomotionInput.JumpPressed && isGrounded)
            {
                _verticalVelocity += _antiBymp + Mathf.Sqrt(jumpSpeed * 3 * gravity);
                _jumpedLastFrame = true;
            }
        }

        private void HandleLateralMovement()
        {
            bool isSprinting = _playerState.CurrrentPlayerMovementState == PlayerMovementState.Sprinting;
            bool isGrounded = _playerState.InGroundedState();


            float lateralAcceleration = !isGrounded ? inAirAcceleration :
                                         isSprinting ? sprintAcceleration : runAcceleration;
            float clampLateralMagnitude = !isGrounded ? sprintSpeed :
                isSprinting ? sprintSpeed : runSpeed;

            Vector3 cameraForwardXZ = new Vector3(_playerCamera.transform.forward.x, 0f, _playerCamera.transform.forward.z).normalized;
            Vector3 cameraRightXZ = new Vector3(_playerCamera.transform.right.x, 0f, _playerCamera.transform.right.z).normalized;
            Vector3 movementDirection = cameraRightXZ * _playerLocomotionInput.MovementInput.x + cameraForwardXZ * _playerLocomotionInput.MovementInput.y;

            Vector3 movementDelta = movementDirection * lateralAcceleration * Time.deltaTime;
            Vector3 newVelocity = _characterController.velocity + movementDelta;

            Vector3 currentDrag = newVelocity.normalized * drag * Time.deltaTime;

            newVelocity = (newVelocity.magnitude > drag * Time.deltaTime) ? newVelocity - currentDrag : Vector3.zero;
            newVelocity = Vector3.ClampMagnitude(new Vector3(newVelocity.x, 0f, newVelocity.z), clampLateralMagnitude);
            newVelocity.y += _verticalVelocity;
            _characterController.Move(newVelocity * Time.deltaTime);
        }

        private void LateUpdate()
        {
            _cameraRotation.x += lookSenseH * _playerLocomotionInput.LookInput.x;
            _cameraRotation.y = Mathf.Clamp(_cameraRotation.y - lookSenseV * _playerLocomotionInput.LookInput.y, -lookLimitV, lookLimitV);

            _playerTargetRotation.x += transform.eulerAngles.x + lookSenseH * _playerLocomotionInput.LookInput.x;
            transform.rotation = Quaternion.Euler(0f, _playerTargetRotation.x, 0f);

            _playerCamera.transform.rotation = Quaternion.Euler(_cameraRotation.y, _cameraRotation.x, 0f);
        }

        private bool IsMovingLaterally()
        {
            Vector3 lateralVelocity = new Vector3(_characterController.velocity.x, 0f, _characterController.velocity.y);

            return lateralVelocity.magnitude > movingThreshold;
        }

        private bool IsGrounded()
        {
            return _playerState.InGroundedState() ? IsGroundedWhileGrounded() : IsGroundedWhileAirborne();
        }

        private bool IsGroundedWhileGrounded()
        {
            Vector3 spherePosition = new Vector3(transform.position.x, transform.position.y - _characterController.radius, transform.position.z);
            return Physics.CheckSphere(spherePosition, _characterController.radius, _groundLayers, QueryTriggerInteraction.Ignore);
        }

        private bool IsGroundedWhileAirborne()
        {
            return _characterController.isGrounded;
        }
    }

}
